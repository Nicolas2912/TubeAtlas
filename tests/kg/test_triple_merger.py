import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np  # type: ignore
import pytest  # type: ignore

from tubeatlas.services.kg.triple_merger import MergeStats, Provenance, Triple, TripleMerger


# ---------------------------------------------------------------------------
# Fixtures & stubs
# ---------------------------------------------------------------------------


def _fake_embed(text: str) -> np.ndarray:  # noqa: D401
    """Very small deterministic embedding for unit tests."""
    if "youtube" in text.lower():  # group similar strings together
        return np.array([1.0, 0.0], dtype=np.float32)
    return np.array([0.0, 1.0], dtype=np.float32)


@pytest.fixture(autouse=True)
def patch_embed(monkeypatch):  # noqa: D401
    """Patch the real embedding function with `_fake_embed`."""
    from tubeatlas.services.kg import triple_merger as tm

    monkeypatch.setattr(tm, "embed_text", _fake_embed, raising=True)
    yield


@pytest.fixture
def merger(tmp_path: Path) -> TripleMerger:  # noqa: D401
    kg_path = tmp_path / "kg.json"
    return TripleMerger(kg_path=kg_path, sim_thr=0.9)


def _prov() -> Provenance:  # noqa: D401
    return Provenance(
        chunk_id="c1",
        video_id="v1",
        channel_id="ch1",
        llm_model="gpt",
        t_created=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_insert_and_exact_duplicate(merger: TripleMerger):  # noqa: D401
    t1 = Triple("Alpha", "rel", "Beta", 0.9, _prov())
    stats1 = merger.merge_batch([t1])
    assert stats1 == MergeStats(inserted=1)

    # exact duplicate – same canonical key
    t_dup = Triple("Alpha", "rel", "Beta", 0.8, _prov())
    stats2 = merger.merge_batch([t_dup])
    assert stats2 == MergeStats(inserted=0, exact_dupes=1)


def test_fuzzy_duplicate(merger: TripleMerger):  # noqa: D401
    # Different strings but mapped to same fake embedding vector -> fuzzy duplicate
    t1 = Triple("YouTube", "is", "Platform", 0.9, _prov())
    t2 = Triple("You Tube", "is", "Platform", 0.8, _prov())  # note space → different signature

    stats1 = merger.merge_batch([t1])
    assert stats1.inserted == 1

    stats2 = merger.merge_batch([t2])
    assert stats2.fuzzy_dupes == 1 and stats2.inserted == 0


def test_new_triple(merger: TripleMerger):  # noqa: D401
    t1 = Triple("Gamma", "rel", "Delta", 0.9, _prov())
    stats = merger.merge_batch([t1])
    assert stats.inserted == 1

    # ensure graph persisted
    with open(merger.kg_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert len(data["links"]) == 1