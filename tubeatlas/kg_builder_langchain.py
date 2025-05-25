"""
Knowledge Graph Builder Module using LangChain.

This module provides functionality to build knowledge graphs from text using LangChain's
LLMGraphTransformer. It supports extracting entities and relationships from text using
OpenAI's language models and storing the results in a structured format.
"""

import os
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple

import sqlite3
import tiktoken
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
        logger.info("Initialized KnowledgeGraphBuilder with %s", 
                   "custom config" if config else "default config")

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
            temperature=self.config.temperature,
            model_name=self.config.model_name
        )

    def _initialize_transformer(self) -> LLMGraphTransformer:
        """
        Initialize the graph transformer.
        
        Returns:
            LLMGraphTransformer: Configured graph transformer instance.
        """
        logger.info("Initializing graph transformer with strict_mode=%s", 
                   self.config.strict_mode)
        return LLMGraphTransformer(
            llm=self.llm,
            strict_mode=self.config.strict_mode,
            additional_instructions=self.config.additional_instructions
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

    def load_text_from_db(self, query: str = "SELECT transcript_text FROM transcripts LIMIT 1") -> str:
        """
        Load text content from the SQLite database.
        
        Args:
            query: SQL query to execute for fetching the text.
            
        Returns:
            str: The fetched text content.
            
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
                    
                text = result[0]
                is_within_limit, token_count = self.check_token_limit(text)
                
                if not is_within_limit:
                    logger.error("Text exceeds token limit: %d tokens (limit: %d)", 
                               token_count, self.config.max_tokens)
                    raise ValueError(f"Text exceeds token limit of {self.config.max_tokens} tokens")
                
                logger.info("Successfully loaded text from database (%d tokens)", token_count)
                return text
                
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
            logger.error("Text exceeds token limit: %d tokens (limit: %d)", 
                        token_count, self.config.max_tokens)
            raise ValueError(f"Text exceeds token limit of {self.config.max_tokens} tokens")
            
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
                    rel.source.id, rel.source.type,
                    rel.type,
                    rel.target.id, rel.target.type
                )
                
                predicate_formatted = rel.type.lower().replace("_", " ")
                output_triples.append({
                    "subject": rel.source.id,
                    "predicate": predicate_formatted,
                    "object": rel.target.id
                })
            
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

def main():
    """Example usage of the KnowledgeGraphBuilder class."""
    try:
        # Create builder with default config
        builder = KnowledgeGraphBuilder()
        
        # Load text from database
        text = builder.load_text_from_db()
        
        # Extract knowledge graph
        graph_data = builder.extract_knowledge_graph(text)
        
        # Save the results
        builder.save_knowledge_graph(graph_data)
        
    except Exception as e:
        logger.error("Error in main execution: %s", e)
        raise

if __name__ == "__main__":
    main()