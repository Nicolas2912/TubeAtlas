import asyncio  # Added for potential async implementation later
import json
import logging
import os
import sqlite3 as sql
import time

import openai
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
DIRECT_API_THRESHOLD = 1_000_000  # Max tokens for a single API call
SLIDING_WINDOW_THRESHOLD = 10_000_000  # Max tokens before recommending Batch API
WINDOW_SIZE = 900_000  # Max tokens per sliding window
OVERLAP_TOKENS = 50_000  # Overlap between windows
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_TEMPERATURE = 0.2

# --- Prompt Templates ---
MAIN_PROMPT_TEMPLATE = """# IDENTITY and PURPOSE

You are an expert content analyst who synthesizes high-level insights from video content. Your goal is to understand the overall themes, patterns, and key messages that emerge across an entire channel's content, rather than focusing on individual videos.

# STEPS

1. First, analyze the overall content to identify:
   - Main themes and recurring topics
   - The creator's core message and philosophy
   - Evolution of ideas across videos
   - Unique perspectives or approaches

2. Then, synthesize these findings into three sections:

CHANNEL OVERVIEW (2-3 sentences):
- A concise summary of the channel's main focus and unique value proposition
- The creator's core philosophy or approach

KEY THEMES (5-7 bullet points):
- Major recurring topics and ideas
- Each theme should be 15-20 words
- Focus on patterns across videos, not individual insights

CORE INSIGHTS (3-5 bullet points):
- The most profound and universal insights that emerge
- Each insight should be 15-20 words
- These should represent the channel's most valuable contributions

# OUTPUT INSTRUCTIONS

- Output all three sections in order
- Use clear, concise language
- Focus on synthesis and patterns, not individual video details
- Each bullet point should be unique and start with different words
- Do not include warnings, notes, or explanations

# INPUT

Input: {input_text}
"""

SYNTHESIS_PROMPT_TEMPLATE = """# FINAL SYNTHESIS

You are an expert content analyst. Below are partial insight summaries from different segments of a YouTube channel. Your task is to synthesize them into a single, cohesive summary, following the same output structure as the initial analysis (CHANNEL OVERVIEW, KEY THEMES, CORE INSIGHTS).

# PARTIAL SUMMARIES

{partial_summaries_text}

# OUTPUT INSTRUCTIONS

- Output CHANNEL OVERVIEW (2-3 sentences)
- Output KEY THEMES (5-7 bullet points, 15-20 words each)
- Output CORE INSIGHTS (3-5 bullet points, 15-20 words each)
- Ensure insights are distinct and represent a holistic view of the channel.
- Do not include warnings, notes, or explanations.
- Synthesize; do not just list or repeat points from partial summaries.
"""

# --- Helper Functions ---


def _count_tokens_simple(text: str) -> int:
    """Roughly estimates token count ny using tiktoken"""
    import tiktoken

    encoding = tiktoken.get_encoding("gpt-4o-mini")
    return len(encoding.encode(text))


def _fetch_transcript_data(db_name: str) -> list:
    """Fetches transcript data (title, text, tokens) from the database."""
    conn = sql.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT title, transcript_text, openai_tokens FROM transcripts WHERE transcript_text IS NOT NULL AND transcript_text != ''"
    )
    rows = cursor.fetchall()
    conn.close()

    transcript_segments = []
    for title, text, tokens_db in rows:
        tokens = tokens_db if tokens_db is not None else _count_tokens_simple(text)
        transcript_segments.append({"title": title, "text": text, "tokens": tokens})
    return transcript_segments


def _prepare_input_string(segments: list) -> str:
    """Formats a list of transcript segments into a single string for the API."""
    formatted_transcripts = [
        f"=== Title: {seg['title']} ===\nTranscript: {seg['text']}" for seg in segments
    ]
    return "\n\n".join(formatted_transcripts)


def _call_openai_api(prompt_content: str) -> str:
    """Makes a call to the OpenAI API and returns the content of the response."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    try:
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt_content}],
            temperature=OPENAI_TEMPERATURE,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise


def _get_insights_direct(transcript_segments: list) -> str:
    """Processes all transcripts in a single API call."""
    logger.info(
        f"Processing {len(transcript_segments)} transcripts directly (single API call)."
    )
    input_text = _prepare_input_string(transcript_segments)
    prompt = MAIN_PROMPT_TEMPLATE.format(input_text=input_text)
    return _call_openai_api(prompt)


async def _get_insights_sliding_window(transcript_segments: list) -> str:
    """Processes transcripts using a sliding window approach if they exceed single call limits."""
    logger.info(
        f"Processing {len(transcript_segments)} transcripts using sliding window."
    )

    windows_data = []
    current_window_segments = []
    current_window_tokens = 0

    segment_idx = 0
    while segment_idx < len(transcript_segments):
        segment = transcript_segments[segment_idx]

        if (
            current_window_tokens + segment["tokens"] > WINDOW_SIZE
            and current_window_segments
        ):
            # Current window is full, process it
            windows_data.append(current_window_segments)

            # Create overlap for the next window
            overlap_calc_tokens = 0
            temp_overlap_segments = []
            for seg_in_overlap in reversed(current_window_segments):
                if overlap_calc_tokens + seg_in_overlap["tokens"] <= OVERLAP_TOKENS:
                    temp_overlap_segments.insert(0, seg_in_overlap)
                    overlap_calc_tokens += seg_in_overlap["tokens"]
                else:
                    break
            current_window_segments = temp_overlap_segments
            current_window_tokens = overlap_calc_tokens
            # Do not increment segment_idx, current segment will be re-evaluated for the new window

        # Add current segment to window if it fits (or if window is empty and segment is too large)
        if (
            not current_window_segments
            or current_window_tokens + segment["tokens"] <= WINDOW_SIZE
        ):
            current_window_segments.append(segment)
            current_window_tokens += segment["tokens"]
            segment_idx += 1
        elif (
            not current_window_segments and segment["tokens"] > WINDOW_SIZE
        ):  # Single segment too large
            logger.warning(
                f"Segment '{segment['title']}' ({segment['tokens']} tokens) is larger than window size ({WINDOW_SIZE} tokens) and will be processed alone."
            )
            windows_data.append([segment])  # Process it in its own window
            current_window_segments = []  # Reset for next
            current_window_tokens = 0
            segment_idx += 1

    if current_window_segments:  # Add any remaining segments as the last window
        windows_data.append(current_window_segments)

    logger.info(f"Segmented into {len(windows_data)} windows.")

    async def process_window(window_segments: list, window_num: int) -> str:
        """Process a single window asynchronously."""
        logger.info(
            f"Processing window {window_num}/{len(windows_data)} with {len(window_segments)} transcripts, {sum(s['tokens'] for s in window_segments)} tokens."
        )
        input_text = _prepare_input_string(window_segments)
        prompt = MAIN_PROMPT_TEMPLATE.format(input_text=input_text)
        try:
            # Assuming _call_openai_api is updated to be async
            insights = await _call_openai_api(prompt)
            return insights
        except Exception as e:
            logger.error(f"Failed to process window {window_num}: {e}")
            return f"Error processing window {window_num}: {e}"

    # Process all windows concurrently
    tasks = [
        process_window(window_segments, i + 1)
        for i, window_segments in enumerate(windows_data)
    ]
    partial_insights_list = await asyncio.gather(*tasks)

    if not partial_insights_list:
        return "No insights could be generated from any window."
    if len(partial_insights_list) == 1:
        return partial_insights_list[0]
    else:
        logger.info("Synthesizing final insights from all partial summaries.")
        partial_summaries_text = "\n\n---\n\n".join(partial_insights_list)
        synthesis_prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            partial_summaries_text=partial_summaries_text
        )
        try:
            # Assuming _call_openai_api is updated to be async
            final_insights = await _call_openai_api(synthesis_prompt)
            return final_insights
        except Exception as e:
            logger.error(f"Failed to synthesize final insights: {e}")
            return (
                "Failed to synthesize final insights. Partial summaries:\n"
                + partial_summaries_text
            )


def _get_insights_batch_api(transcript_segments: list) -> str:
    """Process insights using OpenAI's Batch API for large content volumes.

    Args:
        transcript_segments: List of transcript segments to process

    Returns:
        str: Generated insights from the batch processing
    """
    import json
    import os
    import tempfile

    from openai import OpenAI

    client = OpenAI()
    total_tokens = sum(seg["tokens"] for seg in transcript_segments)
    logger.info(
        f"Total tokens ({total_tokens}) exceed threshold for sliding window. Using Batch API."
    )

    # Prepare batch tasks
    tasks = []
    for idx, segment in enumerate(transcript_segments):
        task = {
            "custom_id": f"segment-{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": OPENAI_MODEL,  # Using latest model
                "temperature": OPENAI_TEMPERATURE,
                "messages": [
                    {
                        "role": "system",
                        "content": MAIN_PROMPT_TEMPLATE.format(
                            input_text=f"Title: {segment['title']}\n\nTranscript: {segment['text']}"
                        ),
                    },
                    {"role": "user", "content": ""},
                ],
            },
        }
        tasks.append(task)

    # Create temporary file for batch tasks
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as temp_file:
        for task in tasks:
            temp_file.write(json.dumps(task) + "\n")
        temp_file_path = temp_file.name

    try:
        # Upload the batch file
        with open(temp_file_path, "rb") as file:
            batch_file = client.files.create(file=file, purpose="batch")

        # Create and submit the batch job
        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        logger.info(f"Batch job created with ID: {batch_job.id}")

        # Wait for completion and retrieve results
        while True:
            batch_job = client.batches.retrieve(batch_job.id)
            if batch_job.status == "completed":
                break
            elif batch_job.status == "failed":
                error_message = "Unknown error"
                if batch_job.errors and batch_job.errors.data:
                    error_message = batch_job.errors.data[0].message
                raise Exception(f"Batch job failed: {error_message}")
            logger.info("Waiting for batch job completion...")
            time.sleep(20)  # Check every 20 seconds

        # Get results
        result_file = client.files.content(batch_job.output_file_id)
        results = []
        for line in result_file.content.decode().splitlines():
            result = json.loads(line)
            results.append(result)

        # Process and combine insights
        insights = []
        for result in sorted(results, key=lambda x: int(x["custom_id"].split("-")[1])):
            if "error" in result:
                logger.error(
                    f"Error in segment {result['custom_id']}: {result['error']}"
                )
                continue
            insight = result["response"]["body"]["choices"][0]["message"]["content"]
            insights.append(insight)

        # Combine insights into final summary
        if not insights:
            return "No insights could be generated from the batch processing."

        # Use a final API call to synthesize insights
        synthesis_prompt = (
            "Synthesize these partial insights into a coherent summary:\n\n"
            + "\n\n".join(insights)
        )
        final_response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYNTHESIS_PROMPT_TEMPLATE},
                {"role": "user", "content": synthesis_prompt},
            ],
        )

        return final_response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return f"Error in batch processing: {str(e)}"
    finally:
        # Cleanup temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


# --- New Helper Function to Get Batch Job Results ---
def get_batch_job_results(batch_id: str) -> dict:
    """
    Retrieves the status and results of a given OpenAI batch job.

    Args:
        batch_id: The ID of the batch job.

    Returns:
        A dictionary containing the batch job's status, and output or error details.
    """
    from openai import OpenAI

    client = OpenAI()  # Assumes OPENAI_API_KEY is set in environment via load_dotenv()
    results_payload = {
        "batch_id": batch_id,
        "status": None,
        "processed_output": [],
        "errors_summary": [],
        "output_file_id": None,
        "error_file_id": None,
        "error_file_raw_content": None,
        "retrieval_message": None,
    }

    try:
        logger.info(f"Retrieving batch job details for ID: {batch_id}")
        batch_job = client.batches.retrieve(batch_id)
        results_payload["status"] = batch_job.status
        results_payload["output_file_id"] = batch_job.output_file_id
        results_payload["error_file_id"] = batch_job.error_file_id

        logger.info(f"Batch job {batch_id} status: {batch_job.status}")

        if batch_job.status == "completed":
            if batch_job.output_file_id:
                logger.info(
                    f"Fetching content from output file: {batch_job.output_file_id}"
                )
                output_file_content_response = client.files.content(
                    batch_job.output_file_id
                )
                output_content_str = output_file_content_response.content.decode(
                    "utf-8"
                )

                parsed_lines = 0
                for line_str in output_content_str.strip().splitlines():
                    if not line_str.strip():
                        continue
                    try:
                        response_json = json.loads(line_str)
                        custom_id = response_json.get("custom_id")

                        # Check for top-level error for the segment (e.g., 429, 500 from OpenAI docs)
                        top_level_error_obj = response_json.get("error")
                        if top_level_error_obj:
                            # Extract nested error if possible, otherwise take the whole object
                            error_detail = top_level_error_obj.get("body", {}).get(
                                "error", top_level_error_obj
                            )
                            results_payload["processed_output"].append(
                                {"custom_id": custom_id, "error": error_detail}
                            )
                            logger.warning(
                                f"Top-level error for batch segment {custom_id}: {error_detail.get('message', str(error_detail))}"
                            )
                            continue

                        response_part = response_json.get("response")
                        if not response_part:
                            results_payload["processed_output"].append(
                                {
                                    "custom_id": custom_id,
                                    "error": {
                                        "message": "Missing 'response' field in batch output line."
                                    },
                                    "raw_segment_response": response_json,
                                }
                            )
                            logger.warning(
                                f"Missing 'response' field for batch segment {custom_id}. Raw: {response_json}"
                            )
                            continue

                        response_body = response_part.get("body", {})
                        # Check for error within the response body (e.g., 400 from OpenAI docs)
                        api_error_in_body = response_body.get("error")
                        if api_error_in_body:
                            results_payload["processed_output"].append(
                                {"custom_id": custom_id, "error": api_error_in_body}
                            )
                            logger.warning(
                                f"API error in response body for batch segment {custom_id}: {api_error_in_body.get('message')}"
                            )
                            continue

                        choices = response_body.get("choices")
                        if choices and isinstance(choices, list) and len(choices) > 0:
                            message = choices[0].get("message")
                            if message and isinstance(message, dict):
                                content = message.get("content")
                                if (
                                    content is not None
                                ):  # Content can be an empty string which is valid
                                    results_payload["processed_output"].append(
                                        {"custom_id": custom_id, "content": content}
                                    )
                                else:  # content key exists but is null
                                    results_payload["processed_output"].append(
                                        {
                                            "custom_id": custom_id,
                                            "warning": "Content is null in message.",
                                            "raw_segment_response": response_json,
                                        }
                                    )
                                    logger.warning(
                                        f"Content is null for batch segment {custom_id}. Raw: {response_json}"
                                    )
                            else:  # message key exists but not a dict, or choices[0] has no 'message'
                                results_payload["processed_output"].append(
                                    {
                                        "custom_id": custom_id,
                                        "warning": "Message structure unexpected or missing in choices.",
                                        "raw_segment_response": response_json,
                                    }
                                )
                                logger.warning(
                                    f"Unexpected message structure for batch segment {custom_id}. Raw: {response_json}"
                                )
                        else:  # 'choices' key missing, not a list, or empty
                            results_payload["processed_output"].append(
                                {
                                    "custom_id": custom_id,
                                    "warning": "Choices structure unexpected or missing in response body.",
                                    "raw_segment_response": response_json,
                                }
                            )
                            logger.warning(
                                f"Unexpected choices structure for batch segment {custom_id}. Raw: {response_json}"
                            )
                        parsed_lines += 1
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Failed to parse JSON line from output file {batch_job.output_file_id}: {e}. Line: '{line_str}'"
                        )
                        results_payload["processed_output"].append(
                            {
                                "custom_id": "unknown_json_error",
                                "error": {
                                    "message": f"JSONDecodeError: {e}",
                                    "line_content": line_str,
                                },
                            }
                        )
                    except (
                        Exception
                    ) as e:  # Catch any other unexpected errors during line processing
                        logger.error(
                            f"Unexpected error processing line from output file {batch_job.output_file_id}: {e}. Line: '{line_str}'",
                            exc_info=True,
                        )
                        results_payload["processed_output"].append(
                            {
                                "custom_id": "unknown_processing_error",
                                "error": {
                                    "message": f"Unexpected error: {e}",
                                    "line_content": line_str,
                                },
                            }
                        )
                logger.info(
                    f"Successfully processed {parsed_lines} lines from output file {batch_job.output_file_id}."
                )
            else:
                results_payload["retrieval_message"] = (
                    "Batch job completed, but no output file ID found."
                )
                logger.warning(results_payload["retrieval_message"])

        elif batch_job.status == "failed":
            if batch_job.errors and batch_job.errors.data:
                for error_obj in batch_job.errors.data:
                    results_payload["errors_summary"].append(
                        {
                            "code": error_obj.code,
                            "message": error_obj.message,
                            "line": getattr(error_obj, "line", None),
                        }
                    )
            else:
                results_payload["errors_summary"].append(
                    {
                        "message": "Batch job failed, but no specific error details found in batch_job.errors.data."
                    }
                )

            if batch_job.error_file_id:
                logger.info(
                    f"Fetching content from error file: {batch_job.error_file_id}"
                )
                try:
                    error_file_content_response = client.files.content(
                        batch_job.error_file_id
                    )
                    results_payload["error_file_raw_content"] = (
                        error_file_content_response.content.decode("utf-8")
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to retrieve or decode error file {batch_job.error_file_id}: {e}",
                        exc_info=True,
                    )
                    # Add to summary as this is an error in retrieving the error file itself
                    results_payload["errors_summary"].append(
                        {
                            "message": f"Failed to retrieve error file content for {batch_job.error_file_id}: {e}"
                        }
                    )
            else:
                logger.info(
                    f"Batch job {batch_id} failed, no separate error file ID provided."
                )
                if not results_payload["errors_summary"] or (
                    len(results_payload["errors_summary"]) == 1
                    and results_payload["errors_summary"][0]["message"].startswith(
                        "Batch job failed, but no specific error details"
                    )
                ):
                    results_payload["errors_summary"].append(
                        {
                            "message": "No error file ID and no further details in batch_job.errors.data."
                        }
                    )
        else:
            results_payload["retrieval_message"] = (
                f"Batch job {batch_id} is currently {batch_job.status}."
            )
            logger.info(results_payload["retrieval_message"])

    except Exception as e:
        logger.error(
            f"Failed to retrieve or process batch job {batch_id}: {e}", exc_info=True
        )
        results_payload["status"] = (
            results_payload["status"] or "error_during_retrieval"
        )  # Preserve status if already set by API
        if results_payload["status"] is None:
            results_payload["status"] = "error_during_retrieval"

        error_message = (
            f"An error occurred while fetching/processing batch job: {str(e)}"
        )
        results_payload["retrieval_message"] = error_message
        # Ensure this top-level error is also in errors_summary
        if not any(
            item.get("message") == error_message
            for item in results_payload["errors_summary"]
        ):
            results_payload["errors_summary"].append(
                {"message": error_message, "code": "retrieval_error"}
            )

    # Construct final, cleaner dictionary
    final_results = {
        "batch_id": results_payload["batch_id"],
        "status": results_payload["status"],
    }

    if results_payload["status"] == "completed":
        final_results["processed_output"] = results_payload["processed_output"]
        if results_payload["output_file_id"]:
            final_results["output_file_id"] = results_payload["output_file_id"]
    elif results_payload["status"] == "failed":
        final_results["errors_summary"] = results_payload["errors_summary"]
        if results_payload["error_file_id"]:
            final_results["error_file_id"] = results_payload["error_file_id"]
        if results_payload["error_file_raw_content"]:  # Only include if it has content
            final_results["error_file_raw_content"] = results_payload[
                "error_file_raw_content"
            ]

    # Include retrieval message if relevant (e.g., for non-terminal states or overarching errors)
    if results_payload["retrieval_message"] and results_payload["status"] not in [
        "completed",
        "failed",
    ]:
        final_results["retrieval_message"] = results_payload["retrieval_message"]
    elif (
        results_payload["status"] == "error_during_retrieval"
        and results_payload["retrieval_message"]
    ):
        final_results["retrieval_message"] = results_payload["retrieval_message"]
        # If retrieval failed, errors_summary should be populated by the except block
        if results_payload["errors_summary"]:
            final_results["errors_summary"] = results_payload["errors_summary"]

    return final_results


# --- Main Dispatcher ---
def get_channel_insights(db_name: str) -> str:
    """
    Fetches insights for a channel, choosing the processing strategy based on total token count.
    """
    logger.info(f"Starting insight generation for channel in DB: {db_name}")
    transcript_segments = _fetch_transcript_data(db_name)

    if not transcript_segments:
        logger.warning("No transcripts found in the database.")
        return "No transcripts found to generate insights."

    total_tokens = sum(seg["tokens"] for seg in transcript_segments)
    logger.info(f"Total estimated tokens for the channel: {total_tokens}")

    if total_tokens == 0:
        logger.warning("Total tokens are zero, no content to process.")
        return "No content to process (total tokens is zero)."

    if total_tokens < DIRECT_API_THRESHOLD:
        logger.info("Strategy: Direct API call (single request).")
        return _get_insights_direct(transcript_segments)
    elif total_tokens < SLIDING_WINDOW_THRESHOLD:
        logger.info("Strategy: Sliding window processing.")
        # Add a note about potential for async implementation
        logger.info(
            "Note: Sliding window is currently synchronous. Consider async for performance on multiple windows."
        )
        return _get_insights_sliding_window(transcript_segments)
    else:
        logger.info("Strategy: Batch API recommended.")
        return _get_insights_batch_api(transcript_segments)


if __name__ == "__main__":
    # Replace 'get_insights' with 'get_channel_insights'
    # insights = get_insights("data/bryanjohnson.db") # Old call
    # insights = get_channel_insights("data/lexfridman.db")
    # print("--- Generated Insights ---")
    # print(insights)
    # print("------------------------")

    # Example: Retrieve results for a specific batch ID
    # Make sure to replace "your_batch_id_here" with an actual ID
    batch_id_to_check = "batch_6834418e37588190b8d09f23c2438a66"
    if batch_id_to_check != "your_batch_id_here":  # Simple check to not run by default
        print(f"--- Results for Batch ID: {batch_id_to_check} ---")
        batch_results = get_batch_job_results(batch_id_to_check)
        import json  # For pretty printing

        print(json.dumps(batch_results, indent=2))
        print("------------------------------------")
    pass
