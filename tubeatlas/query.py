#!/usr/bin/env python3
"""
Database query tool for YouTube transcript databases.

This script provides predefined queries for analyzing YouTube transcript databases.
Modify the QUERIES dictionary to add or change queries.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Adjust Pandas Display Options ---
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)  # Limit number of rows displayed
pd.set_option('display.max_colwidth', 100)  # Limit column width

# Predefined queries - modify these to add your own queries
QUERIES = {
    'all_videos': """
        SELECT 
            video_id,
            title,
            publish_date,
            length_seconds,
            category_name,
            gemini_tokens,
            openai_tokens
        FROM transcripts
        ORDER BY publish_date DESC
    """,
    
    'video_stats': """
        SELECT 
            COUNT(*) as total_videos,
            SUM(CASE WHEN transcript_text IS NULL OR transcript_text = '' THEN 0 ELSE 1 END) as videos_with_transcripts,
            ROUND(SUM(CASE WHEN transcript_text IS NULL OR transcript_text = '' THEN 0 ELSE 1 END) * 100.0 / COUNT(*), 1) as transcript_coverage_percent,
            SUM(length_seconds) as total_duration_seconds,
            ROUND(AVG(length_seconds), 1) as avg_duration_seconds,
            SUM(gemini_tokens) as total_gemini_tokens,
            SUM(openai_tokens) as total_openai_tokens,
            COUNT(DISTINCT category_name) as unique_categories
        FROM transcripts
    """,
    
    'category_stats': """
        SELECT 
            category_name,
            COUNT(*) as video_count,
            AVG(length_seconds) as avg_duration_seconds,
            SUM(gemini_tokens) as total_gemini_tokens
        FROM transcripts
        GROUP BY category_name
        ORDER BY video_count DESC
    """,
    
    'longest_videos': """
        SELECT 
            video_id,
            title,
            publish_date,
            length_seconds,
            category_name
        FROM transcripts
        ORDER BY length_seconds DESC
        LIMIT 10
    """,
    
    'recent_videos': """
        SELECT 
            video_id,
            title,
            publish_date,
            length_seconds,
            category_name
        FROM transcripts
        ORDER BY publish_date DESC
        LIMIT 10
    """
}

def get_db_path(channel_name: str) -> Path:
    """
    Get the database path for a channel.
    
    Args:
        channel_name: Name of the channel (without .db extension)
        
    Returns:
        Path object for the database file
    """
    data_dir = Path(__file__).parent.parent / "data"
    return data_dir / f"{channel_name}.db"

def check_database(db_path: Path) -> tuple[bool, Optional[str]]:
    """
    Check if database exists and has the required table.
    
    Args:
        db_path: Path to the database file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not db_path.exists():
        return False, f"Database file not found: {db_path}"
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if transcripts table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='transcripts'
        """)
        
        if not cursor.fetchone():
            return False, f"Database exists but 'transcripts' table not found in {db_path}"
            
        # Check table structure
        cursor.execute("PRAGMA table_info(transcripts)")
        columns = {row[1] for row in cursor.fetchall()}
        required_columns = {
            'video_id', 'title', 'publish_date', 'length_seconds',
            'description', 'tags', 'category_name', 'transcript_text'
        }
        
        missing_columns = required_columns - columns
        if missing_columns:
            return False, f"Database table is missing required columns: {missing_columns}"
            
        return True, None
        
    except sqlite3.Error as e:
        return False, f"Database error: {e}"
    finally:
        if 'conn' in locals():
            conn.close()

def query_transcript_db(db_path: Path, query: str) -> pd.DataFrame:
    """
    Execute a query on the transcript database.
    
    Args:
        db_path: Path to the database file
        query: SQL query to execute
        
    Returns:
        DataFrame containing query results
        
    Raises:
        ValueError: If database is invalid or query fails
    """
    # Validate database first
    is_valid, error_msg = check_database(db_path)
    if not is_valid:
        raise ValueError(error_msg)
        
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(query, conn)
        return df
    except sqlite3.Error as e:
        raise ValueError(f"Database error executing query: {e}")
    except Exception as e:
        raise ValueError(f"Error executing query: {e}")
    finally:
        if conn:
            conn.close()

def list_available_databases() -> Dict[str, bool]:
    """
    List all available transcript databases.
    
    Returns:
        Dictionary mapping database names to their validity status
    """
    data_dir = Path(__file__).parent.parent / "data"
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return {}
        
    dbs = list(data_dir.glob("*.db"))
    if not dbs:
        logger.error("No database files found in data directory")
        return {}
        
    logger.info("Available databases:")
    db_status = {}
    for db in sorted(dbs):
        is_valid, error_msg = check_database(db)
        status = "✓" if is_valid else "✗"
        logger.info(f"{status} {db.stem}")
        if not is_valid:
            logger.error(f"  Error: {error_msg}")
        db_status[db.stem] = is_valid
    
    return db_status

def run_queries(channel_name: str) -> None:
    """
    Run all predefined queries for a channel.
    
    Args:
        channel_name: Name of the channel (without .db extension)
    """
    db_path = get_db_path(channel_name)
    
    try:
        # First check if database exists and is valid
        is_valid, error_msg = check_database(db_path)
        if not is_valid:
            logger.error(error_msg)
            return
            
        logger.info(f"\nRunning queries for channel: {channel_name}")
        logger.info("=" * 80)
        
        # Run each query
        for query_name, query in QUERIES.items():
            logger.info(f"\nQuery: {query_name}")
            logger.info("-" * 40)
            
            df = query_transcript_db(db_path, query)
            if df.empty:
                logger.info("No results found")
            else:
                print(f"\nResults ({len(df)} rows):")
                print(df)
                
    except Exception as e:
        logger.error(f"Error running queries: {e}")

def main():
    """Main entry point."""
    # List available databases
    db_status = list_available_databases()
    
    if not db_status:
        logger.error("No valid databases found. Please run the download script first.")
        return
        
    # Run queries for each valid database
    for channel_name, is_valid in db_status.items():
        if is_valid:
            run_queries(channel_name)

if __name__ == "__main__":
    main()
