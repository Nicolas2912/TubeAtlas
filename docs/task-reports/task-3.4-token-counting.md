# Task 3.4: Integrate Token Counting into TranscriptService

Date: 2024-06-27

## Task Description

The primary goal of this task was to integrate token counting into the `TranscriptService`. This involved creating a token counting utility, updating the `TranscriptService` to use it, and ensuring the new functionality was thoroughly tested.

## Implementation Details

The implementation was broken down into two main parts: creating the utility and integrating it.

### 1. Token Counting Utility

A new utility file was created at `src/tubeatlas/utils/token_counter.py`.

-   **`count_tokens(text: str, model: str) -> int`**: This function serves as the single point of truth for tokenization. It uses the `tiktoken` library to accurately count the number of tokens in a given string based on the specified model (e.g., "gpt-4").
-   **Model Support**: It includes a dictionary to map common model names to their respective `tiktoken` encodings and has a fallback mechanism to let `tiktoken` attempt to find an encoding for models not explicitly listed.
-   **Error Handling**: It raises a `KeyError` if a model is not supported by `tiktoken`, providing clear feedback.

### 2. Integration with `TranscriptService`

The `TranscriptService` at `src/tubeatlas/services/transcript_service.py` was updated as follows:

-   **`TypedDict` Updates**: The `TranscriptSegment` and `Transcript` `TypedDict`s were extended to include new fields:
    -   `TranscriptSegment`: Added `token_count: int`.
    -   `Transcript`: Added `total_token_count: Optional[int]`.
-   **`get_transcript` Enhancement**: The core transcript fetching method was modified to:
    1.  Loop through each segment of a successfully fetched transcript.
    2.  Call the new `count_tokens_util` for each segment's text.
    3.  Store the result in the segment's `token_count` field.
    4.  Sum the segment counts to calculate and store the `total_token_count` for the entire transcript.
-   **`count_tokens` Method**: The existing placeholder method in `TranscriptService` was updated to be a simple wrapper around the new `count_tokens_util`, ensuring consistent token counting logic across the application.

## Verification

To validate the implementation, new unit tests were created and existing ones were updated.

-   **`tests/unit/test_token_counter.py`**: A new test file was created to verify the `count_tokens` utility. It includes tests for:
    -   Correct token counts for standard models.
    -   Proper error handling for unsupported models.
    -   Correct handling of empty strings.
-   **`tests/unit/test_transcript_service.py`**:
    -   A new test, `test_get_transcript_success_with_token_counts`, was added to specifically verify that `token_count` and `total_token_count` are calculated and added correctly. This test uses a mock of the `count_tokens_util` to ensure predictable behavior.
    -   Existing tests were updated to reflect the new structure of the `TranscriptSegment` dictionary (i.e., the presence of the `token_count` field), ensuring the entire test suite remains consistent and passes.

All 50 unit tests in the suite passed, confirming that the new functionality works as expected and has not introduced any regressions.
