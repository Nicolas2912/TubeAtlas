import json
import logging
import os
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import openai
import spacy
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


class HybridKGBuilder:
    def __init__(self, openai_api_key: str = None):
        # Load spaCy model for classical NLP
        self.nlp = spacy.load("en_core_web_trf")

        # Initialize OpenAI client (modern format)
        self.openai_client = None
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)

        # Initialize graph
        self.graph = nx.MultiDiGraph()

        # Entity and relation stores
        self.entities = defaultdict(set)
        self.relations = []

        # Track extraction statistics
        self.extraction_stats = {
            "total_entities_classical": 0,
            "total_relations_classical": 0,
            "total_entities_llm": 0,
            "total_relations_llm": 0,
            "chunks_processed": 0,
        }

        # Dynamic entity and relation quality metrics
        self.entity_metrics = {
            "min_length": 2,
            "max_length": 50,
            "min_word_count": 1,
            "max_word_count": 5,
            "min_confidence": 0.6,
        }

        self.relation_metrics = {
            "min_length": 2,
            "max_length": 30,
            "min_confidence": 0.7,
            "min_semantic_similarity": 0.3,  # For detecting redundant relations
        }

        # Track entity frequencies for quality assessment
        self.entity_frequencies = defaultdict(int)
        self.relation_frequencies = defaultdict(int)

        # Track entity co-occurrences for relation quality
        self.entity_co_occurrences = defaultdict(int)

    def preprocess_transcript(
        self, transcript: str, max_chars: int = 2000
    ) -> List[str]:
        """Clean and chunk transcript for processing, limiting to first 2000 chars (whole sentences only)"""
        # Remove timestamps and clean text
        cleaned = re.sub(r"\[\d+:\d+\]", "", transcript)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Process with spaCy
        doc = self.nlp(cleaned)
        sentences = [sent.text for sent in doc.sents]

        # Accumulate sentences up to 2000 chars, but do not cut mid-sentence
        selected_sentences = []
        total_len = 0
        for sent in sentences:
            sent_len = len(sent)
            if total_len + sent_len > max_chars:
                break
            selected_sentences.append(sent)
            total_len += sent_len

        if not selected_sentences:
            logger.warning("Transcript is too short or no sentences found.")
            return []

        # Create overlapping chunks for context
        chunks = []
        window_size = 4
        step = window_size // 2 if window_size > 1 else 1
        for i in range(0, len(selected_sentences), step):
            chunk = " ".join(selected_sentences[i : i + window_size])
            if len(chunk) > 50:  # Skip very short chunks
                chunks.append(chunk)

        logger.info(
            f"Created {len(chunks)} chunks from {len(selected_sentences)} sentences (max 2000 chars)"
        )
        return chunks

    def extract_classical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using spaCy NER with improved filtering"""
        doc = self.nlp(text)
        entities = defaultdict(list)

        for ent in doc.ents:
            # More lenient filtering - only skip very short or clearly irrelevant
            if len(ent.text.strip()) > 1 and ent.label_ not in ["CARDINAL"]:
                entities[ent.label_].append(
                    {
                        "text": ent.text.strip(),
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "label": ent.label_,
                    }
                )

        # Add noun phrases as potential entities
        for chunk in doc.noun_chunks:
            if len(chunk.text.strip()) > 2 and chunk.text.strip() not in [
                e["text"] for ents in entities.values() for e in ents
            ]:
                entities["NOUN_PHRASE"].append(
                    {
                        "text": chunk.text.strip(),
                        "start": chunk.start_char,
                        "end": chunk.end_char,
                        "label": "NOUN_PHRASE",
                    }
                )

        entity_count = sum(len(ents) for ents in entities.values())
        logger.info(
            f"Classical extraction found {entity_count} entities: {dict([(k, len(v)) for k, v in entities.items()])}"
        )
        self.extraction_stats["total_entities_classical"] += entity_count

        return dict(entities)

    def extract_classical_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract relations using improved dependency parsing patterns"""
        doc = self.nlp(text)
        relations = []

        for sent in doc.sents:
            # Pattern 1: Subject-Verb-Object
            for token in sent:
                if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                    subject = self._get_entity_span(token)
                    verb = token.head.text

                    # Find direct object
                    for child in token.head.children:
                        if child.dep_ in ["dobj", "pobj", "attr"]:
                            obj = self._get_entity_span(child)
                            relations.append((subject, verb, obj))

            # Pattern 2: Prepositional relations (X preposition Y)
            for token in sent:
                if token.pos_ == "ADP":  # Preposition
                    if token.head.pos_ in ["NOUN", "PROPN"] and token.nbor(1).pos_ in [
                        "NOUN",
                        "PROPN",
                    ]:
                        head_entity = self._get_entity_span(token.head)
                        prep_entity = self._get_entity_span(token.nbor(1))
                        relations.append((head_entity, token.text, prep_entity))

            # Pattern 3: Appositive relations (X, Y -> X is Y)
            for token in sent:
                if token.dep_ == "appos":
                    head_entity = self._get_entity_span(token.head)
                    appos_entity = self._get_entity_span(token)
                    relations.append((head_entity, "is", appos_entity))

            # Pattern 4: Compound relations
            for token in sent:
                if token.dep_ == "compound":
                    compound = self._get_entity_span(token)
                    head = self._get_entity_span(token.head)
                    relations.append((compound, "part_of", head))

        # Add co-occurrence relations for entities in same sentence
        entities_in_sent = []
        for sent in doc.sents:
            sent_entities = []
            for ent in doc.ents:
                if sent.start <= ent.start < sent.end:
                    sent_entities.append(ent.text.strip())

            # Create co-occurrence relations between entities in same sentence
            for i, ent1 in enumerate(sent_entities):
                for ent2 in sent_entities[i + 1 :]:
                    if ent1 != ent2:
                        relations.append((ent1, "co_occurs_with", ent2))

        # Filter out very short or empty relations
        relations = [
            (s, r, o)
            for s, r, o in relations
            if len(s.strip()) > 1 and len(o.strip()) > 1 and s.strip() != o.strip()
        ]

        logger.info(f"Classical extraction found {len(relations)} relations")
        self.extraction_stats["total_relations_classical"] += len(relations)

        # Log first few relations for debugging
        for i, (s, r, o) in enumerate(relations[:5]):
            logger.info(f"  Relation {i+1}: '{s}' --[{r}]--> '{o}'")

        return relations

    def _get_entity_span(self, token) -> str:
        """Get the full entity span for a token (including compounds, etc.)"""
        # Start with the token itself
        start = token.i
        end = token.i + 1

        # Extend left for compounds
        for child in token.children:
            if child.dep_ in ["compound", "amod"] and child.i < token.i:
                start = min(start, child.i)

        # Extend right for compounds
        for child in token.children:
            if child.dep_ in ["compound"] and child.i > token.i:
                end = max(end, child.i + 1)

        # Get the text span
        span_text = token.doc[start:end].text
        return span_text.strip()

    def enhance_with_llm(self, text: str, classical_entities: Dict) -> Dict:
        """Use LLM to enhance and validate classical extraction"""
        if not self.openai_client:
            logger.warning("No OpenAI client available, skipping LLM enhancement")
            return {
                "validated_entities": classical_entities,
                "new_entities": {},
                "relationships": [],
                "corrections": [],
            }

        # Create prompt for entity enhancement with stronger JSON formatting instructions
        entities_str = json.dumps(
            {k: [e["text"] for e in v] for k, v in classical_entities.items()}, indent=2
        )

        prompt = f"""You are a precise JSON generator. Your task is to analyze text and extract entities and relationships.
        You MUST respond with ONLY a valid JSON object, no other text or explanation.

        Given this text and the entities extracted by classical NLP, enhance and validate:

        Text: {text}...

        Classical entities found: {entities_str}

        Instructions:
        1. Validate existing entities (keep the good ones)
        2. Add missing important entities (people, organizations, concepts, etc.)
        3. Extract meaningful relationships between entities
        4. Provide confidence scores (0-1) for relationships

        Your response must be a single JSON object with exactly these fields:
        {{
            "validated_entities": {{"type": ["entity1", "entity2"]}},
            "new_entities": {{"type": ["new_entity1", "new_entity2"]}},
            "relationships": [["entity1", "relation", "entity2", confidence_score]],
            "corrections": ["any issues with classical extraction"]
        }}

        Remember: Respond with ONLY the JSON object, no other text."""

        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",  # Using gpt-4 as it's more reliable
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise JSON generator. Always respond with valid JSON only.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=1000,
                    temperature=0.1,
                )

                # Get the response content
                content = response.choices[0].message.content.strip()

                # Log the raw response for debugging
                logger.debug(f"Raw LLM response: {content}")

                # Validate response is not empty
                if not content:
                    raise ValueError("Empty response from LLM")

                # Try to parse JSON
                try:
                    result = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing failed: {e}")
                    logger.error(f"Invalid JSON content: {content}")
                    raise

                # Validate required fields
                required_fields = [
                    "validated_entities",
                    "new_entities",
                    "relationships",
                    "corrections",
                ]
                if not all(field in result for field in required_fields):
                    raise ValueError(
                        f"Missing required fields in response. Got: {list(result.keys())}"
                    )

                # Log LLM results
                llm_entity_count = sum(
                    len(ents) for ents in result.get("new_entities", {}).values()
                )
                llm_relation_count = len(result.get("relationships", []))
                logger.info(
                    f"LLM enhancement found {llm_entity_count} new entities and {llm_relation_count} relationships"
                )

                self.extraction_stats["total_entities_llm"] += llm_entity_count
                self.extraction_stats["total_relations_llm"] += llm_relation_count

                return result

            except Exception as e:
                logger.error(
                    f"LLM enhancement attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                )
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(
                        "All retry attempts failed, falling back to classical extraction only"
                    )
                    return {
                        "validated_entities": classical_entities,
                        "new_entities": {},
                        "relationships": [],
                        "corrections": [],
                    }

    def assess_entity_quality(
        self, entity: str, context: str = None
    ) -> Tuple[bool, float]:
        """Dynamically assess entity quality based on linguistic and statistical features"""
        # Basic cleaning
        entity = str(entity).strip()
        if not entity:
            return False, 0.0

        # Process with spaCy
        doc = self.nlp(entity)

        # Check linguistic features
        if len(doc) == 0:
            return False, 0.0

        # Calculate quality score based on multiple factors
        quality_score = 0.0
        factors = []

        # 1. Length and word count checks
        if (
            self.entity_metrics["min_length"]
            <= len(entity)
            <= self.entity_metrics["max_length"]
        ):
            factors.append(1.0)
        else:
            factors.append(0.0)

        word_count = len([token for token in doc if not token.is_punct])
        if (
            self.entity_metrics["min_word_count"]
            <= word_count
            <= self.entity_metrics["max_word_count"]
        ):
            factors.append(1.0)
        else:
            factors.append(0.0)

        # 2. Linguistic quality checks
        # Check if it's a proper noun or noun phrase
        is_proper_noun = any(token.pos_ == "PROPN" for token in doc)
        is_noun_phrase = any(token.dep_ in ["nsubj", "dobj", "pobj"] for token in doc)
        if is_proper_noun or is_noun_phrase:
            factors.append(1.0)
        else:
            factors.append(0.5)

        # 3. Check for pronouns and temporal words using spaCy
        is_pronoun = any(token.pos_ == "PRON" for token in doc)
        is_temporal = any(
            token.ent_type_ == "DATE" or token.ent_type_ == "TIME" for token in doc
        )
        if is_pronoun or is_temporal:
            factors.append(0.0)
        else:
            factors.append(1.0)

        # 4. Check entity frequency in the corpus
        freq_score = min(
            1.0, self.entity_frequencies[entity.lower()] / 10
        )  # Normalize frequency
        factors.append(1.0 - freq_score)  # Prefer less frequent entities

        # 5. Check if entity appears in context (if provided)
        if context:
            context_doc = self.nlp(context)
            context_entities = [ent.text.lower() for ent in context_doc.ents]
            if entity.lower() in context_entities:
                factors.append(1.0)
            else:
                factors.append(0.5)

        # Calculate final quality score
        quality_score = sum(factors) / len(factors)

        # Update entity frequency
        self.entity_frequencies[entity.lower()] += 1

        return quality_score >= self.entity_metrics["min_confidence"], quality_score

    def assess_relation_quality(
        self, subject: str, relation: str, object: str, context: str = None
    ) -> Tuple[bool, float]:
        """Dynamically assess relation quality based on linguistic and statistical features"""
        # Basic cleaning
        subject = str(subject).strip()
        relation = str(relation).strip()
        object = str(object).strip()

        if not all([subject, relation, object]):
            return False, 0.0

        # Process with spaCy
        subj_doc = self.nlp(subject)
        rel_doc = self.nlp(relation)
        obj_doc = self.nlp(object)

        # Calculate quality score based on multiple factors
        quality_score = 0.0
        factors = []

        # 1. Length checks
        if (
            self.relation_metrics["min_length"]
            <= len(relation)
            <= self.relation_metrics["max_length"]
        ):
            factors.append(1.0)
        else:
            factors.append(0.0)

        # 2. Linguistic quality checks
        # Check if relation is a verb or preposition
        is_verb = any(token.pos_ == "VERB" for token in rel_doc)
        is_prep = any(token.pos_ == "ADP" for token in rel_doc)
        if is_verb or is_prep:
            factors.append(1.0)
        else:
            factors.append(0.5)

        # 3. Check for generic relations
        is_generic = any(
            token.lemma_ in ["be", "have", "do", "make", "get"] for token in rel_doc
        )
        if is_generic:
            factors.append(0.3)
        else:
            factors.append(1.0)

        # 4. Check relation frequency
        rel_key = f"{subject.lower()}_{relation.lower()}_{object.lower()}"
        freq_score = min(
            1.0, self.relation_frequencies[rel_key] / 5
        )  # Normalize frequency
        factors.append(1.0 - freq_score)  # Prefer less frequent relations

        # 5. Check semantic similarity between subject and object
        if subject.lower() != object.lower():  # Avoid self-relations
            subj_vec = subj_doc.vector
            obj_vec = obj_doc.vector
            if np.any(subj_vec) and np.any(obj_vec):  # Check if vectors are not zero
                similarity = np.dot(subj_vec, obj_vec) / (
                    np.linalg.norm(subj_vec) * np.linalg.norm(obj_vec)
                )
                if similarity >= self.relation_metrics["min_semantic_similarity"]:
                    factors.append(1.0)
                else:
                    factors.append(0.5)
            else:
                factors.append(0.5)
        else:
            factors.append(0.0)

        # 6. Check if relation appears in context (if provided)
        if context:
            context_doc = self.nlp(context)
            context_text = context.lower()
            if all(
                term.lower() in context_text for term in [subject, relation, object]
            ):
                factors.append(1.0)
            else:
                factors.append(0.5)

        # Calculate final quality score
        quality_score = sum(factors) / len(factors)

        # Update relation frequency
        self.relation_frequencies[rel_key] += 1

        # Update co-occurrence statistics
        self.entity_co_occurrences[(subject.lower(), object.lower())] += 1

        return quality_score >= self.relation_metrics["min_confidence"], quality_score

    def clean_entity(self, entity: str, context: str = None) -> Optional[str]:
        """Clean and validate an entity using dynamic quality assessment"""
        # Basic cleaning
        entity = str(entity).strip()
        if not entity:
            return None

        # Assess entity quality
        is_valid, quality_score = self.assess_entity_quality(entity, context)

        if is_valid:
            # Process with spaCy for proper casing
            doc = self.nlp(entity)
            cleaned = " ".join(token.text for token in doc)
            return cleaned

        return None

    def clean_relation(
        self,
        relation: str,
        subject: str = None,
        object: str = None,
        context: str = None,
    ) -> Optional[str]:
        """Clean and validate a relation using dynamic quality assessment"""
        # Basic cleaning
        relation = str(relation).strip()
        if not relation:
            return None

        # If we have subject and object, assess the full relation
        if subject and object:
            is_valid, quality_score = self.assess_relation_quality(
                subject, relation, object, context
            )
            if not is_valid:
                return None

        # Process with spaCy for proper casing
        doc = self.nlp(relation)
        cleaned = " ".join(token.text for token in doc)

        return cleaned

    def deduplicate_relations(
        self, relations: List[Tuple[str, str, str]], context: str = None
    ) -> List[Tuple[str, str, str]]:
        """Remove duplicate and redundant relations using semantic similarity"""
        seen = set()
        unique_relations = []

        for subj, rel, obj in relations:
            # Clean entities and relation
            clean_subj = self.clean_entity(subj, context)
            clean_obj = self.clean_entity(obj, context)
            clean_rel = self.clean_relation(rel, clean_subj, clean_obj, context)

            if not all([clean_subj, clean_obj, clean_rel]):
                continue

            # Create a unique key for the relation
            rel_key = (clean_subj, clean_rel, clean_obj)

            # Skip if we've seen this relation before
            if rel_key in seen:
                continue

            # Check semantic similarity with existing relations
            is_similar = False
            for existing in seen:
                ex_subj, ex_rel, ex_obj = existing
                # Check if relations are semantically similar
                if (
                    self.are_entities_similar(clean_subj, ex_subj)
                    and self.are_entities_similar(clean_obj, ex_obj)
                    and self.are_relations_similar(clean_rel, ex_rel)
                ):
                    is_similar = True
                    break

            if not is_similar:
                seen.add(rel_key)
                unique_relations.append((clean_subj, clean_rel, clean_obj))

        return unique_relations

    def are_entities_similar(self, ent1: str, ent2: str) -> bool:
        """Check if two entities are semantically similar"""
        doc1 = self.nlp(ent1)
        doc2 = self.nlp(ent2)

        if not np.any(doc1.vector) or not np.any(doc2.vector):
            return ent1.lower() == ent2.lower()

        similarity = np.dot(doc1.vector, doc2.vector) / (
            np.linalg.norm(doc1.vector) * np.linalg.norm(doc2.vector)
        )
        return similarity > 0.8  # High threshold for entity similarity

    def are_relations_similar(self, rel1: str, rel2: str) -> bool:
        """Check if two relations are semantically similar"""
        doc1 = self.nlp(rel1)
        doc2 = self.nlp(rel2)

        if not np.any(doc1.vector) or not np.any(doc2.vector):
            return rel1.lower() == rel2.lower()

        similarity = np.dot(doc1.vector, doc2.vector) / (
            np.linalg.norm(doc1.vector) * np.linalg.norm(doc2.vector)
        )
        return similarity > 0.7  # Slightly lower threshold for relation similarity

    def merge_extractions(
        self,
        classical_entities: Dict,
        classical_relations: List[Tuple],
        llm_enhanced: Dict,
    ) -> Dict:
        """Intelligently merge classical and LLM results with improved filtering"""
        merged = {"entities": defaultdict(set), "relations": []}

        # Process classical entities with context
        context = ""  # We could pass the actual chunk text here if available
        for ent_type, entities in classical_entities.items():
            for entity in entities:
                if isinstance(entity, dict):
                    entity_text = entity["text"]
                else:
                    entity_text = entity
                cleaned_entity = self.clean_entity(entity_text, context)
                if cleaned_entity:
                    merged["entities"][ent_type].add(cleaned_entity)

        # Process LLM entities
        for ent_type, entities in llm_enhanced.get("new_entities", {}).items():
            for entity in entities:
                cleaned_entity = self.clean_entity(entity, context)
                if cleaned_entity:
                    merged["entities"][ent_type].add(cleaned_entity)

        # Process classical relations with context
        classical_relations = self.deduplicate_relations(classical_relations, context)
        merged["relations"].extend(classical_relations)

        # Process LLM relations with confidence check
        for rel in llm_enhanced.get("relationships", []):
            if len(rel) >= 4 and rel[3] > self.relation_metrics["min_confidence"]:
                subj, rel_type, obj = rel[0], rel[1], rel[2]
                cleaned_subj = self.clean_entity(subj, context)
                cleaned_obj = self.clean_entity(obj, context)
                cleaned_rel = self.clean_relation(
                    rel_type, cleaned_subj, cleaned_obj, context
                )

                if all([cleaned_subj, cleaned_obj, cleaned_rel]):
                    merged["relations"].append((cleaned_subj, cleaned_rel, cleaned_obj))

        # Final deduplication of all relations
        merged["relations"] = self.deduplicate_relations(merged["relations"], context)

        logger.info(
            f"Merged results: {sum(len(ents) for ents in merged['entities'].values())} entities, {len(merged['relations'])} relations"
        )

        return merged

    def build_graph_from_chunk(self, chunk: str) -> None:
        """Process a single text chunk and add to graph"""
        logger.info(f"Processing chunk: {chunk[:100]}...")

        # Step 1: Classical extraction
        classical_entities = self.extract_classical_entities(chunk)
        classical_relations = self.extract_classical_relations(chunk)

        # Step 2: LLM enhancement
        llm_results = self.enhance_with_llm(chunk, classical_entities)

        # Step 3: Merge results
        merged = self.merge_extractions(
            classical_entities, classical_relations, llm_results
        )

        # Step 4: Add to graph
        # Add entities as nodes
        nodes_added = 0
        for ent_type, entities in merged["entities"].items():
            for entity in entities:
                if entity not in self.graph.nodes:
                    self.graph.add_node(entity, type=ent_type)
                    nodes_added += 1

        # Add relations as edges
        edges_added = 0
        for subj, rel, obj in merged["relations"]:
            # Add nodes if they don't exist
            if subj not in self.graph.nodes:
                self.graph.add_node(subj, type="INFERRED")
            if obj not in self.graph.nodes:
                self.graph.add_node(obj, type="INFERRED")

            # Add edge
            self.graph.add_edge(subj, obj, relation=rel)
            edges_added += 1

        logger.info(f"Added {nodes_added} new nodes and {edges_added} edges to graph")
        self.extraction_stats["chunks_processed"] += 1

    def process_transcript(
        self, transcript: str, max_chars: int = 2000
    ) -> nx.MultiDiGraph:
        """Main method to process entire transcript"""
        logger.info("Starting transcript processing...")
        logger.info("Preprocessing transcript...")
        chunks = self.preprocess_transcript(transcript, max_chars=max_chars)

        logger.info(f"Processing {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            self.build_graph_from_chunk(chunk)

        logger.info("Post-processing graph...")
        self.post_process_graph()

        # Log final statistics
        stats = self.get_statistics()
        logger.info(f"Final graph: {stats['nodes']} nodes, {stats['edges']} edges")
        logger.info(f"Extraction summary: {self.extraction_stats}")

        return self.graph

    def post_process_graph(self) -> None:
        """Clean and optimize the final graph"""
        initial_nodes = self.graph.number_of_nodes()

        # Remove nodes with very low degree (likely noise) - but be less aggressive
        low_degree_nodes = [node for node, degree in self.graph.degree() if degree == 0]
        nodes_to_remove = low_degree_nodes[
            : len(low_degree_nodes) // 3
        ]  # Remove only 1/3 of isolated nodes
        self.graph.remove_nodes_from(nodes_to_remove)

        logger.info(
            f"Post-processing: removed {len(nodes_to_remove)} isolated nodes ({initial_nodes} -> {self.graph.number_of_nodes()})"
        )

    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        if self.graph.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0, "avg_degree": 0, "connected_components": 0}

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "avg_degree": sum(dict(self.graph.degree()).values())
            / self.graph.number_of_nodes(),
            "connected_components": nx.number_weakly_connected_components(self.graph),
        }

    def print_graph_details(self):
        """Print detailed information about the graph"""
        logger.info("\n=== GRAPH DETAILS ===")
        logger.info(f"Nodes ({self.graph.number_of_nodes()}):")
        for node, data in self.graph.nodes(data=True):
            logger.info(f"  - {node} ({data.get('type', 'UNKNOWN')})")

        logger.info(f"\nEdges ({self.graph.number_of_edges()}):")
        for source, target, data in self.graph.edges(data=True):
            logger.info(
                f"  - '{source}' --[{data.get('relation', 'related_to')}]--> '{target}'"
            )

    def export_graph(self, format: str = "gexf", filename: str = "knowledge_graph"):
        """Export graph in various formats"""
        if format == "gexf":
            nx.write_gexf(self.graph, f"{filename}.gexf")
        elif format == "json":
            data = nx.node_link_data(self.graph)
            with open(f"{filename}.json", "w") as f:
                json.dump(data, f, indent=2)
        elif format == "graphml":
            nx.write_graphml(self.graph, f"{filename}.graphml")
        elif format == "triples_json":
            # Convert graph to triples format
            triples = []
            for source, target, data in self.graph.edges(data=True):
                triple = {
                    "subject": source,
                    "predicate": data.get("relation", "related_to"),
                    "object": target,
                }
                triples.append(triple)

            # Create the final JSON structure
            output_data = {"triples": triples}

            # Save to file
            with open(f"{filename}.json", "w") as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Knowledge graph exported as triples to {filename}.json")


# Usage example
if __name__ == "__main__":
    # Initialize builder
    kg_builder = HybridKGBuilder(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Sample transcript
    sample_transcript = """
    [00:01] Welcome to today's discussion about artificial intelligence and machine learning.
    Today we have Dr. Sarah Johnson from MIT talking about neural networks.
    Dr. Johnson has been working on deep learning for over 10 years.
    She recently published research on transformer architectures in Nature magazine.
    The research shows how attention mechanisms can improve natural language processing tasks.
    """

    # Build knowledge graph
    graph = kg_builder.process_transcript(sample_transcript, max_chars=2000)

    # Print detailed graph information
    kg_builder.print_graph_details()

    # Get statistics
    stats = kg_builder.get_statistics()
    print(
        f"\nFinal Result: Graph built with {stats['nodes']} nodes and {stats['edges']} edges"
    )

    # Export graph in triples format
    kg_builder.export_graph("triples_json", "data/example_kg")
