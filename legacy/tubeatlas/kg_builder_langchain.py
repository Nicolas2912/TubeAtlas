"""
Knowledge Graph Builder Module using LangChain.

This module provides functionality to build knowledge graphs from text using LangChain's
LLMGraphTransformer. It supports extracting entities and relationships from text using
OpenAI's language models and storing the results in a structured format.
"""

import asyncio
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import tiktoken
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from langchain_openai import ChatOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class GraphBuilderConfig:
    """Configuration settings for the Knowledge Graph Builder."""

    model_name: str = "gpt-4.1-mini"
    temperature: float = 0.0
    strict_mode: bool = True
    db_path: str = "data/AndrejKarpathy.db"
    output_path: str = "data/example_kg_langchain.json"
    additional_instructions: str = "Focus on direct factual relationships. Identify specific research topics and fields of study."
    max_tokens: int = 1_000_000  # Maximum number of tokens allowed in input text


class KnowledgeGraphBuilder:
    """
    A class for building knowledge graphs from text using LangChain's LLMGraphTransformer.

    This class provides methods to extract entities and relationships from text using
    OpenAI's language models and transform them into a structured knowledge graph format.

    Attributes:
        config (GraphBuilderConfig): Configuration settings for the graph builder
        llm (ChatOpenAI): The language model instance
        transformer (LLMGraphTransformer): The graph transformer instance
        tokenizer: The tokenizer instance for counting tokens
    """

    def __init__(self, config: Optional[GraphBuilderConfig] = None):
        """
        Initialize the KnowledgeGraphBuilder.

        Args:
            config: Optional configuration settings. If None, default settings will be used.

        Raises:
            ValueError: If OpenAI API key is not set in environment variables.
        """
        self.config = config or GraphBuilderConfig()
        self._validate_environment()
        self.llm = self._initialize_llm()
        self.transformer = self._initialize_transformer()
        self.tokenizer = self._initialize_tokenizer()
        logger.info(
            "Initialized KnowledgeGraphBuilder with %s",
            "custom config" if config else "default config",
        )

    def _validate_environment(self) -> None:
        """
        Validate the environment setup.

        Raises:
            ValueError: If OpenAI API key is not set.
        """
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OpenAI API key not found in environment variables")
            raise ValueError("Please set your OPENAI_API_KEY environment variable")

    def _initialize_llm(self) -> ChatOpenAI:
        """
        Initialize the language model.

        Returns:
            ChatOpenAI: Configured language model instance.
        """
        logger.info("Initializing language model: %s", self.config.model_name)
        return ChatOpenAI(
            temperature=self.config.temperature, model_name=self.config.model_name
        )

    def _initialize_transformer(self) -> LLMGraphTransformer:
        """
        Initialize the graph transformer.

        Returns:
            LLMGraphTransformer: Configured graph transformer instance.
        """
        logger.info(
            "Initializing graph transformer with strict_mode=%s",
            self.config.strict_mode,
        )
        return LLMGraphTransformer(
            llm=self.llm,
            strict_mode=self.config.strict_mode,
            additional_instructions=self.config.additional_instructions,
        )

    def _initialize_tokenizer(self) -> tiktoken.Encoding:
        """
        Initialize the tokenizer for the specified model.

        Returns:
            tiktoken.Encoding: The tokenizer instance.

        Raises:
            ValueError: If the model name is not supported.
        """
        try:
            # Map model names to their tokenizer names
            model_to_tokenizer = {
                "gpt-4.1-mini": "cl100k_base",  # GPT-4 uses cl100k_base
                "gpt-3.5-turbo-0125": "cl100k_base",  # GPT-3.5 Turbo also uses cl100k_base
            }

            tokenizer_name = model_to_tokenizer.get(self.config.model_name)
            if not tokenizer_name:
                raise ValueError(f"Unsupported model: {self.config.model_name}")

            return tiktoken.get_encoding(tokenizer_name)

        except Exception as e:
            logger.error("Error initializing tokenizer: %s", e)
            raise

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Args:
            text: The text to count tokens for.

        Returns:
            int: The number of tokens in the text.
        """
        return len(self.tokenizer.encode(text))

    def check_token_limit(self, text: str) -> Tuple[bool, int]:
        """
        Check if the text exceeds the token limit.

        Args:
            text: The text to check.

        Returns:
            Tuple[bool, int]: (is_within_limit, token_count)
        """
        token_count = self.count_tokens(text)
        logger.info("Token count: %d", token_count)
        is_within_limit = token_count <= self.config.max_tokens
        return is_within_limit, token_count

    def load_text_from_db(
        self,
        query: str = "SELECT title, transcript_text FROM transcripts WHERE openai_tokens > 2000 LIMIT 1",
    ) -> str:
        """
        Load text content from the SQLite database.

        Args:
            query: SQL query to execute for fetching the text.

        Returns:
            str: The fetched transcript text content.

        Raises:
            sqlite3.Error: If there's an error accessing the database.
            ValueError: If the text exceeds the token limit.
        """
        try:
            logger.info("Loading text from database: %s", self.config.db_path)
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                result = cursor.fetchone()

                if not result:
                    logger.warning("No text found in database with query: %s", query)
                    return ""

                title, transcript_text = result
                print(f"Processing transcript: {title}")

                is_within_limit, token_count = self.check_token_limit(transcript_text)

                if not is_within_limit:
                    logger.error(
                        "Text exceeds token limit: %d tokens (limit: %d)",
                        token_count,
                        self.config.max_tokens,
                    )
                    raise ValueError(
                        f"Text exceeds token limit of {self.config.max_tokens} tokens"
                    )

                logger.info(
                    "Successfully loaded text from database (%d tokens)", token_count
                )
                return transcript_text

        except sqlite3.Error as e:
            logger.error("Database error: %s", e)
            raise

    def load_text_from_string(self, text: str) -> str:
        """
        Use provided text content directly.

        Args:
            text: The text content to use.

        Returns:
            str: The provided text content.

        Raises:
            ValueError: If the text exceeds the token limit.
        """
        is_within_limit, token_count = self.check_token_limit(text)

        if not is_within_limit:
            logger.error(
                "Text exceeds token limit: %d tokens (limit: %d)",
                token_count,
                self.config.max_tokens,
            )
            raise ValueError(
                f"Text exceeds token limit of {self.config.max_tokens} tokens"
            )

        logger.info("Using provided text content (%d tokens)", token_count)
        return text

    def extract_knowledge_graph(self, text: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Extract knowledge graph from text using the LLM transformer.

        Args:
            text: The input text to process.

        Returns:
            Dict containing the extracted triples.

        Raises:
            ValueError: If the text is empty or no graph documents are generated.
        """
        if not text.strip():
            logger.error("Empty text provided for knowledge graph extraction")
            raise ValueError("Text content cannot be empty")

        try:
            logger.info("Creating document from text")
            doc = Document(page_content=text)

            logger.info("Converting document to graph documents")
            graph_documents = self.transformer.convert_to_graph_documents([doc])

            if not graph_documents:
                logger.error("No graph documents generated from text")
                raise ValueError("Failed to generate graph documents from text")

            graph_doc = graph_documents[0]
            output_triples = []

            # Log extracted nodes
            logger.info("Extracted %d nodes", len(graph_doc.nodes))
            for node in graph_doc.nodes:
                logger.debug("Node: ID=%s, Type=%s", node.id, node.type)

            # Log and process relationships
            logger.info("Extracted %d relationships", len(graph_doc.relationships))
            for rel in graph_doc.relationships:
                logger.debug(
                    "Relationship: %s (%s) -> %s -> %s (%s)",
                    rel.source.id,
                    rel.source.type,
                    rel.type,
                    rel.target.id,
                    rel.target.type,
                )

                predicate_formatted = rel.type.lower().replace("_", " ")
                output_triples.append(
                    {
                        "subject": rel.source.id,
                        "predicate": predicate_formatted,
                        "object": rel.target.id,
                    }
                )

            return {"triples": output_triples}

        except Exception as e:
            logger.error("Error extracting knowledge graph: %s", e)
            raise

    def save_knowledge_graph(self, graph_data: Dict[str, List[Dict[str, str]]]) -> None:
        """
        Save the knowledge graph to a JSON file.

        Args:
            graph_data: The knowledge graph data to save.

        Raises:
            IOError: If there's an error writing to the file.
        """
        try:
            output_path = Path(self.config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(graph_data, f, indent=2)

            logger.info("Knowledge graph saved to %s", output_path)

        except IOError as e:
            logger.error("Error saving knowledge graph: %s", e)
            raise

    def _calc_api_costs(self):
        # 1. get all number of tokens for one channel
        conn = sqlite3.connect(self.config.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(openai_tokens) FROM transcripts")
        result = cursor.fetchone()
        total_tokens = result[0]

        # 2. get the cost of the tokens
        pricing_dict_per_million_tokens = {
            "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
            "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
            "gpt-4.1-nano": {"input": 0.100, "cached_input": 0.025, "output": 0.400},
        }

        # 3. calculate the cost
        input_token_costs = round(
            pricing_dict_per_million_tokens[self.config.model_name]["input"]
            * total_tokens
            / 1000000,
            3,
        )

        logger.info("Total tokens: %d", total_tokens)
        logger.info(
            f"Input token costs for model {self.config.model_name}: {input_token_costs}"
        )
        return total_tokens, input_token_costs

    def build_complete_knowledge_graph(
        self, batch_size: int = 3
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Build a knowledge graph from all transcripts in the database (synchronous version).

        This method loads all transcript texts from the database, processes them in batches
        to avoid token limits, and combines the results into a single knowledge graph.

        Args:
            batch_size: Number of transcripts to process in each batch to manage token limits.

        Returns:
            Dict containing the combined knowledge graph triples from all transcripts.

        Raises:
            sqlite3.Error: If there's an error accessing the database.
            ValueError: If no transcripts are found or processing fails.
        """
        try:
            logger.info("Building complete knowledge graph from all transcripts")

            # Load all transcripts from database
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT title, transcript_text FROM transcripts WHERE transcript_text IS NOT NULL AND transcript_text != ''"
                )
                all_transcripts = cursor.fetchall()

                if not all_transcripts:
                    logger.warning("No transcripts found in database")
                    raise ValueError("No transcripts found in database")

                logger.info("Found %d transcripts to process", len(all_transcripts))

            # Calculate total number of batches
            total_batches = (len(all_transcripts) + batch_size - 1) // batch_size
            logger.info(
                "Will process in %d batches of size %d", total_batches, batch_size
            )

            # Combine all triples from batches
            all_triples = []
            processed_count = 0

            # Process transcripts in batches to manage token limits
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(all_transcripts))
                batch = all_transcripts[start_idx:end_idx]

                logger.info(
                    "Processing batch %d/%d (transcripts %d-%d)",
                    batch_num + 1,
                    total_batches,
                    start_idx + 1,
                    end_idx,
                )

                # Combine texts from current batch
                batch_text = ""
                batch_titles = []

                for title, transcript_text in batch:
                    batch_titles.append(title)
                    # Add title as context and transcript text
                    batch_text += f"{transcript_text}\n ---\n"

                # Check token limit for batch
                is_within_limit, token_count = self.check_token_limit(batch_text)

                if not is_within_limit:
                    logger.warning(
                        "Batch exceeds token limit (%d tokens), reducing batch size",
                        token_count,
                    )
                    # Process transcripts individually if batch is too large
                    for title, transcript_text in batch:
                        individual_text = f"=== {title} ===\n{transcript_text}"
                        (
                            is_individual_within_limit,
                            individual_token_count,
                        ) = self.check_token_limit(individual_text)

                        if is_individual_within_limit:
                            logger.info(
                                "Processing individual transcript: %s (%d tokens)",
                                title,
                                individual_token_count,
                            )
                            try:
                                graph_data = self.extract_knowledge_graph(
                                    individual_text
                                )
                                all_triples.extend(graph_data["triples"])
                                processed_count += 1
                            except Exception as e:
                                logger.error(
                                    "Error processing transcript '%s': %s", title, e
                                )
                        else:
                            logger.warning(
                                "Skipping transcript '%s' - exceeds token limit (%d tokens)",
                                title,
                                individual_token_count,
                            )
                else:
                    # Process entire batch
                    logger.info(
                        "Processing batch with %d transcripts (%d tokens)",
                        len(batch),
                        token_count,
                    )
                    try:
                        graph_data = self.extract_knowledge_graph(batch_text)
                        all_triples.extend(graph_data["triples"])
                        processed_count += len(batch)
                    except Exception as e:
                        logger.error("Error processing batch: %s", e)
                        # Fallback to individual processing
                        for title, transcript_text in batch:
                            individual_text = f"=== {title} ===\n{transcript_text}"
                            try:
                                graph_data = self.extract_knowledge_graph(
                                    individual_text
                                )
                                all_triples.extend(graph_data["triples"])
                                processed_count += 1
                            except Exception as e:
                                logger.error(
                                    "Error processing transcript '%s': %s", title, e
                                )

            logger.info(
                "Successfully processed %d/%d transcripts",
                processed_count,
                len(all_transcripts),
            )
            logger.info("Extracted %d total triples", len(all_triples))

            # Remove duplicate triples while preserving order
            unique_triples = []
            seen_triples = set()

            for triple in all_triples:
                triple_key = (triple["subject"], triple["predicate"], triple["object"])
                if triple_key not in seen_triples:
                    seen_triples.add(triple_key)
                    unique_triples.append(triple)

            logger.info("After deduplication: %d unique triples", len(unique_triples))

            return {"triples": unique_triples}

        except sqlite3.Error as e:
            logger.error("Database error: %s", e)
            raise
        except Exception as e:
            logger.error("Error building complete knowledge graph: %s", e)
            raise

    async def build_complete_knowledge_graph_async(
        self, batch_size: int = 3, max_concurrent: int = 5
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Build a knowledge graph from all transcripts in the database (asynchronous version).

        This method loads all transcript texts from the database, processes them concurrently
        using async operations to speed up the knowledge graph building process.

        Args:
            batch_size: Number of transcripts to process in each batch to manage token limits.
            max_concurrent: Maximum number of concurrent batch processing operations.

        Returns:
            Dict containing the combined knowledge graph triples from all transcripts.

        Raises:
            sqlite3.Error: If there's an error accessing the database.
            ValueError: If no transcripts are found or processing fails.
        """
        try:
            logger.info(
                "Building complete knowledge graph from all transcripts (async)"
            )

            # Load all transcripts from database
            with sqlite3.connect(self.config.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT title, transcript_text FROM transcripts WHERE transcript_text IS NOT NULL AND transcript_text != ''"
                )
                all_transcripts = cursor.fetchall()

                if not all_transcripts:
                    logger.warning("No transcripts found in database")
                    raise ValueError("No transcripts found in database")

                logger.info("Found %d transcripts to process", len(all_transcripts))

            # Calculate total number of batches
            total_batches = (len(all_transcripts) + batch_size - 1) // batch_size
            logger.info(
                "Will process in %d batches of size %d with max %d concurrent operations",
                total_batches,
                batch_size,
                max_concurrent,
            )

            # Create batches
            batches = []
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(all_transcripts))
                batch = all_transcripts[start_idx:end_idx]
                batches.append((batch_num + 1, batch))

            # Process batches concurrently with semaphore to limit concurrent operations
            semaphore = asyncio.Semaphore(max_concurrent)
            all_triples = []

            async def process_batch(batch_info):
                batch_num, batch = batch_info
                async with semaphore:
                    return await self._process_batch_async(
                        batch_num, batch, total_batches
                    )

            # Execute all batch processing concurrently
            logger.info("Starting concurrent batch processing...")
            batch_results = await asyncio.gather(
                *[process_batch(batch_info) for batch_info in batches],
                return_exceptions=True,
            )

            # Collect results and handle exceptions
            processed_count = 0
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error("Error processing batch %d: %s", i + 1, result)
                else:
                    triples, count = result
                    all_triples.extend(triples)
                    processed_count += count

            logger.info("Successfully processed %d transcripts", processed_count)
            logger.info("Extracted %d total triples", len(all_triples))

            # Remove duplicate triples while preserving order
            unique_triples = []
            seen_triples = set()

            for triple in all_triples:
                triple_key = (triple["subject"], triple["predicate"], triple["object"])
                if triple_key not in seen_triples:
                    seen_triples.add(triple_key)
                    unique_triples.append(triple)

            logger.info("After deduplication: %d unique triples", len(unique_triples))

            return {"triples": unique_triples}

        except sqlite3.Error as e:
            logger.error("Database error: %s", e)
            raise
        except Exception as e:
            logger.error("Error building complete knowledge graph (async): %s", e)
            raise

    async def _process_batch_async(
        self, batch_num: int, batch: List[Tuple[str, str]], total_batches: int
    ) -> Tuple[List[Dict[str, str]], int]:
        """
        Process a single batch of transcripts asynchronously.

        Args:
            batch_num: The batch number for logging.
            batch: List of (title, transcript_text) tuples.
            total_batches: Total number of batches for logging.

        Returns:
            Tuple of (triples_list, processed_count).
        """
        try:
            start_idx = (batch_num - 1) * len(batch) + 1
            end_idx = start_idx + len(batch) - 1

            logger.info(
                "Processing batch %d/%d (transcripts %d-%d)",
                batch_num,
                total_batches,
                start_idx,
                end_idx,
            )

            # Combine texts from current batch
            batch_text = ""
            for title, transcript_text in batch:
                batch_text += f"{transcript_text}\n ---\n"

            # Check token limit for batch
            is_within_limit, token_count = self.check_token_limit(batch_text)

            if not is_within_limit:
                logger.warning(
                    "Batch %d exceeds token limit (%d tokens), processing individually",
                    batch_num,
                    token_count,
                )
                # Process transcripts individually if batch is too large
                all_triples = []
                processed_count = 0

                for title, transcript_text in batch:
                    individual_text = f"=== {title} ===\n{transcript_text}"
                    (
                        is_individual_within_limit,
                        individual_token_count,
                    ) = self.check_token_limit(individual_text)

                    if is_individual_within_limit:
                        logger.info(
                            "Processing individual transcript: %s (%d tokens)",
                            title,
                            individual_token_count,
                        )
                        try:
                            doc = Document(page_content=individual_text)
                            graph_documents = (
                                await self.transformer.aconvert_to_graph_documents(
                                    [doc]
                                )
                            )

                            if graph_documents:
                                graph_doc = graph_documents[0]
                                for rel in graph_doc.relationships:
                                    predicate_formatted = rel.type.lower().replace(
                                        "_", " "
                                    )
                                    all_triples.append(
                                        {
                                            "subject": rel.source.id,
                                            "predicate": predicate_formatted,
                                            "object": rel.target.id,
                                        }
                                    )
                                processed_count += 1
                        except Exception as e:
                            logger.error(
                                "Error processing transcript '%s': %s", title, e
                            )
                    else:
                        logger.warning(
                            "Skipping transcript '%s' - exceeds token limit (%d tokens)",
                            title,
                            individual_token_count,
                        )

                return all_triples, processed_count
            else:
                # Process entire batch
                logger.info(
                    "Processing batch %d with %d transcripts (%d tokens)",
                    batch_num,
                    len(batch),
                    token_count,
                )
                try:
                    doc = Document(page_content=batch_text)
                    graph_documents = (
                        await self.transformer.aconvert_to_graph_documents([doc])
                    )

                    if not graph_documents:
                        logger.error(
                            "No graph documents generated from batch %d", batch_num
                        )
                        return [], 0

                    graph_doc = graph_documents[0]
                    output_triples = []

                    for rel in graph_doc.relationships:
                        predicate_formatted = rel.type.lower().replace("_", " ")
                        output_triples.append(
                            {
                                "subject": rel.source.id,
                                "predicate": predicate_formatted,
                                "object": rel.target.id,
                            }
                        )

                    return output_triples, len(batch)

                except Exception as e:
                    logger.error("Error processing batch %d: %s", batch_num, e)
                    return [], 0

        except Exception as e:
            logger.error("Error in batch processing %d: %s", batch_num, e)
            return [], 0


async def main_async():
    """Example usage of the KnowledgeGraphBuilder class with async processing."""
    try:
        # make config
        config = GraphBuilderConfig(
            model_name="gpt-4.1-mini",
            temperature=0.0,
            strict_mode=True,
            db_path="data/bryanjohnson.db",
            output_path="data/complete_kg_langchain_bryanjohnson_async.json",
        )
        # Create builder with default config
        builder = KnowledgeGraphBuilder(config)

        # Build complete knowledge graph from all transcripts (async version)
        # Process with batch_size=25 and max 3 concurrent operations
        graph_data = await builder.build_complete_knowledge_graph_async(
            batch_size=25, max_concurrent=3
        )

        # Save the results
        builder.save_knowledge_graph(graph_data)

        # Calculate API costs
        builder._calc_api_costs()

    except Exception as e:
        logger.error("Error in async main execution: %s", e)
        raise


def main():
    """Example usage of the KnowledgeGraphBuilder class."""
    try:
        # make config
        config = GraphBuilderConfig(
            model_name="gpt-4.1-mini",
            temperature=0.0,
            strict_mode=True,
            db_path="data/bryanjohnson.db",
            output_path="data/complete_kg_langchain_bryanjohnson.json",
        )
        # Create builder with default config
        builder = KnowledgeGraphBuilder(config)

        # Option 1: Load single text from database
        # text = builder.load_text_from_db()
        # graph_data = builder.extract_knowledge_graph(text)

        # Option 2: Build complete knowledge graph from all transcripts (sync version)
        # graph_data = builder.build_complete_knowledge_graph(batch_size=25)

        # Option 3: Use async version for faster processing
        graph_data = asyncio.run(
            builder.build_complete_knowledge_graph_async(
                batch_size=10, max_concurrent=3
            )
        )

        # Save the results
        builder.save_knowledge_graph(graph_data)

        # Calculate API costs
        builder._calc_api_costs()

    except Exception as e:
        logger.error("Error in main execution: %s", e)
        raise


if __name__ == "__main__":
    main()
