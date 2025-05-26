#!/usr/bin/env python3
"""
YouTube Transcript Manager

This module provides functionality to download and manage YouTube video transcripts
and metadata. It supports both file-based and SQLite storage options.

Features:
- Download transcripts from YouTube channels
- Fetch video metadata (title, description, tags, etc.)
- Store data in files or SQLite database
- Token counting for transcripts
- Category mapping support
"""

# Standard library imports
import argparse
import glob
import logging
import os
import re
import sqlite3
import sys
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Third-party imports
try:
    from google import genai
except ImportError:
    genai = None
try:
    import scrapetube
except ImportError:
    scrapetube = None
from tqdm import tqdm
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    build = None
    HttpError = Exception
try:
    import isodate
except ImportError:
    isodate = None
try:
    import openai
except ImportError:
    openai = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "transcripts")
DEFAULT_REGION_CODE = 'US'
DEFAULT_BATCH_SIZE = 50
DEFAULT_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

class YouTubeTranscriptManager:
    """
    Manages downloading and storing YouTube transcripts and metadata.
    
    This class handles the complete workflow of fetching video transcripts,
    metadata, and storing them either in files or a SQLite database.
    
    Attributes:
        youtube_service: Initialized YouTube Data API service object
        output_dir (str): Directory for file storage
        storage_type (str): Storage type ('file' or 'sqlite')
        db_path (str): Path for SQLite database
        category_mapping (dict): Mapping of category IDs to names
        conn: SQLite database connection
        cursor: SQLite database cursor
    """

    def __init__(
        self,
        youtube_service,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        storage_type: str = 'file',
        db_path: str = 'transcripts.db',
        region_code: str = DEFAULT_REGION_CODE
    ):
        """
        Initialize the transcript manager.

        Args:
            youtube_service: Initialized YouTube Data API service object
            output_dir: Directory for file storage
            storage_type: Storage type ('file' or 'sqlite')
            db_path: Path for SQLite database
            region_code: Region code for category mapping (e.g., 'US', 'GB')

        Raises:
            ValueError: If youtube_service is not provided or storage_type is invalid
        """
        if not youtube_service:
            raise ValueError("YouTube Data API service object is required.")
        
        self.youtube_service = youtube_service
        self.output_dir = output_dir
        self.storage_type = storage_type.lower()
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.category_mapping = {}

        if self.storage_type not in ['file', 'sqlite']:
            raise ValueError("storage_type must be either 'file' or 'sqlite'")

        self._setup_storage()
        self._fetch_category_mapping(region_code)

    def _setup_storage(self) -> None:
        """Set up storage based on the selected storage type."""
        if self.storage_type == 'file':
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Using file storage in directory: {self.output_dir}")
        elif self.storage_type == 'sqlite':
            self._initialize_db()

    def _fetch_category_mapping(self, region_code: str = DEFAULT_REGION_CODE) -> None:
        """
        Fetch YouTube category mapping for the specified region.

        Args:
            region_code: Region code for category mapping (e.g., 'US', 'GB')
        """
        logger.info(f"Fetching YouTube category mapping for region: {region_code}")
        try:
            request = self.youtube_service.videoCategories().list(
                part="snippet",
                regionCode=region_code
            )
            response = request.execute()
            self.category_mapping = {
                item['id']: item['snippet']['title']
                for item in response.get('items', [])
            }
            logger.info(f"Successfully fetched {len(self.category_mapping)} categories")
        except HttpError as e:
            logger.error(f"API error fetching categories: {e}")
            self.category_mapping = {}
        except Exception as e:
            logger.error(f"Unexpected error fetching categories: {e}")
            self.category_mapping = {}

    def _initialize_db(self) -> None:
        """
        Initialize SQLite database and create necessary tables.
        
        Creates a transcripts table with all required columns if it doesn't exist.
        Also adds new columns (gemini_tokens, openai_tokens) if they don't exist.
        
        Raises:
            sqlite3.Error: If database initialization fails
        """
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Create main table with all columns
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS transcripts (
                    video_id TEXT PRIMARY KEY,
                    title TEXT,
                    publish_date DATE,
                    length_seconds INTEGER,
                    description TEXT,
                    tags TEXT,
                    category_name TEXT,
                    kg_graph TEXT,
                    transcript_text TEXT,
                    gemini_tokens INTEGER,
                    openai_tokens INTEGER,
                    download_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Add new columns if they don't exist
            for column in ['gemini_tokens', 'openai_tokens']:
                try:
                    self.cursor.execute(f'ALTER TABLE transcripts ADD COLUMN {column} INTEGER')
                except sqlite3.OperationalError:
                    pass  # Column already exists
                    
            self.conn.commit()
            logger.info(f"Connected to SQLite database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to or initializing SQLite database: {e}")
            self.conn = None
            self.cursor = None

    def close_db(self) -> None:
        """Close the SQLite database connection if it's open."""
        if self.conn:
            self.conn.close()
            logger.info("SQLite database connection closed")
            self.conn = None
            self.cursor = None

    def __del__(self):
        """Ensure database connection is closed when the object is destroyed."""
        self.close_db()

    def _get_channel_video_ids(self, channel_url: str) -> List[str]:
        """
        Get video IDs from a YouTube channel URL using the YouTube Data API.
        
        Args:
            channel_url: URL of the YouTube channel
            
        Returns:
            List of video IDs found in the channel
            
        Raises:
            Exception: If API fails to fetch video IDs
        """
        logger.info(f"Getting video IDs from channel URL: {channel_url}")
        try:
            # First, get the channel ID from the URL
            channel_name = get_channel_name_from_url(channel_url)
            logger.info(f"Channel name/ID extracted: {channel_name}")
            
            # If we got a channel ID directly (starts with UC), use it
            if channel_name.startswith('UC'):
                channel_id = channel_name
            else:
                # Otherwise, search for the channel to get its ID
                logger.info("Searching for channel ID...")
                search_response = self.youtube_service.search().list(
                    q=channel_name,
                    type='channel',
                    part='id',
                    maxResults=1
                ).execute()
                
                if not search_response.get('items'):
                    raise ValueError(f"Could not find channel ID for {channel_name}")
                    
                channel_id = search_response['items'][0]['id']['channelId']
                logger.info(f"Found channel ID: {channel_id}")
            
            # Now get all videos using the channel ID
            logger.info("Fetching all videos from channel...")
            video_ids = []
            next_page_token = None
            
            while True:
                # Get uploads playlist ID (all videos are in the uploads playlist)
                if not video_ids:  # Only need to do this once
                    channel_response = self.youtube_service.channels().list(
                        part='contentDetails',
                        id=channel_id
                    ).execute()
                    
                    if not channel_response.get('items'):
                        raise ValueError(f"Could not get channel details for {channel_id}")
                        
                    uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
                    logger.info(f"Found uploads playlist ID: {uploads_playlist_id}")
                
                # Get videos from the uploads playlist
                playlist_response = self.youtube_service.playlistItems().list(
                    part='contentDetails',
                    playlistId=uploads_playlist_id,
                    maxResults=50,  # Maximum allowed by the API
                    pageToken=next_page_token
                ).execute()
                
                # Add video IDs from this page
                new_video_ids = [item['contentDetails']['videoId'] for item in playlist_response.get('items', [])]
                video_ids.extend(new_video_ids)
                
                # Check if there are more pages
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token:
                    break
                    
                logger.info(f"Fetched {len(video_ids)} videos so far...")
            
            if not video_ids:
                logger.error("No videos found. This might be due to:")
                logger.error("1. The channel has no public videos")
                logger.error("2. The channel's videos are not accessible")
                return []
                
            logger.info(f"Found {len(video_ids)} total video IDs")
            
            # Log some additional information about the videos
            if video_ids:
                logger.info("First 5 video IDs found:")
                for vid in video_ids[:5]:
                    logger.info(f"  - {vid}")
                logger.info("Last 5 video IDs found:")
                for vid in video_ids[-5:]:
                    logger.info(f"  - {vid}")
                    
                # Try to get video details for the first video to verify access
                try:
                    request = self.youtube_service.videos().list(
                        part="snippet",
                        id=video_ids[0]
                    ).execute()
                    if request.get('items'):
                        logger.info(f"Successfully verified access to first video: {request['items'][0]['snippet']['title']}")
                except Exception as e:
                    logger.warning(f"Could not verify video access: {e}")
            
            return video_ids
            
        except Exception as e:
            logger.error(f"Error getting video IDs for URL {channel_url}: {e}")
            logger.error("This might be due to:")
            logger.error("1. Network connectivity issues")
            logger.error("2. YouTube API rate limiting")
            logger.error("3. Channel access restrictions")
            logger.error("4. Invalid channel URL format")
            return []

    def _parse_duration(self, duration_str: str) -> int:
        """
        Parse ISO 8601 duration string to total seconds.
        
        Args:
            duration_str: ISO 8601 duration string (e.g., 'PT1H30M15S')
            
        Returns:
            Total duration in seconds
        """
        if not duration_str:
            return 0
            
        if isodate is None:
            # Fallback to regex parsing if isodate is not available
            m = re.match(r'^PT(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?$', duration_str)
            if not m:
                return 0
            hours = int(m.group('h') or 0)
            minutes = int(m.group('m') or 0)
            seconds = int(m.group('s') or 0)
            return hours * 3600 + minutes * 60 + seconds
            
        try:
            duration = isodate.parse_duration(duration_str)
            return int(duration.total_seconds())
        except Exception as e:
            logger.warning(f"Error parsing duration '{duration_str}': {e}")
            return 0

    def _fetch_metadata_batch(self, video_ids_batch: List[str]) -> Dict[str, dict]:
        """
        Fetch metadata for a batch of video IDs.
        
        Args:
            video_ids_batch: List of video IDs to fetch metadata for
            
        Returns:
            Dictionary mapping video IDs to their metadata
        """
        metadata = {}
        if not self.youtube_service:
            logger.error("YouTube service not initialized")
            return metadata

        try:
            request = self.youtube_service.videos().list(
                part="snippet,contentDetails",
                id=",".join(video_ids_batch)
            )
            response = request.execute()

            for item in response.get('items', []):
                video_id = item['id']
                snippet = item.get('snippet', {})
                content_details = item.get('contentDetails', {})

                # Extract and parse metadata
                title = snippet.get('title', 'N/A')
                description = snippet.get('description', '')
                tags_list = snippet.get('tags', [])
                category_id = snippet.get('categoryId')
                category_name = self.category_mapping.get(category_id, 'Unknown') if category_id else 'Unknown'

                # Parse dates and duration
                published_at_str = snippet.get('publishedAt')
                publish_date = None
                if published_at_str:
                    try:
                        publish_date = datetime.fromisoformat(
                            published_at_str.replace('Z', '+00:00')
                        ).strftime('%Y-%m-%d')
                    except ValueError as e:
                        logger.warning(f"Could not parse date '{published_at_str}' for video {video_id}: {e}")

                duration_iso = content_details.get('duration')
                length_seconds = self._parse_duration(duration_iso)

                metadata[video_id] = {
                    'title': title,
                    'publish_date': publish_date,
                    'description': description,
                    'length_seconds': length_seconds,
                    'tags': tags_list,
                    'category_name': category_name
                }
                
            logger.debug(f"Successfully fetched metadata for {len(metadata)} videos")
            
        except HttpError as e:
            logger.error(f"API error during metadata fetch: {e}")
            if e.resp.status == 403:
                logger.error("Quota possibly exceeded. Check Google Cloud Console")
        except Exception as e:
            logger.error(f"Unexpected error during metadata fetch: {e}")

        return metadata

    def _get_transcript(self, video_id: str) -> Optional[str]:
        """
        Fetch transcript text for a single video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Transcript text if available, None otherwise
        """
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            text = "\n".join(entry['text'] for entry in transcript_list)
            return text.strip()
        except Exception as e:
            logger.warning(f"Could not get transcript for {video_id}: {e}")
            return None

    def _count_tokens(self, content: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
        """
        Count tokens using both Gemini and OpenAI models.
        
        Args:
            content: Text content to count tokens for
            
        Returns:
            Tuple of (gemini_tokens, openai_tokens)
        """
        gemini_tokens = None
        openai_tokens = None

        if not content:
            return None, None

        # Count tokens with Gemini
        if genai:
            try:
                client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                response = client.models.count_tokens(
                    model='gemini-2.0-flash-001',
                    contents=content
                )
                gemini_tokens = response.total_tokens
            except Exception as e:
                logger.warning(f"Error counting tokens with Gemini: {e}")

        # Count tokens with OpenAI
        if openai:
            try:
                import tiktoken
                encoding = tiktoken.encoding_for_model("gpt-4o")
                openai_tokens = len(encoding.encode(content))
            except Exception as e:
                logger.warning(f"Error counting tokens with OpenAI: {e}")

        return gemini_tokens, openai_tokens

    def _save_data(self, video_id: str, data: dict) -> bool:
        """
        Save transcript and metadata to storage.
        
        Args:
            video_id: YouTube video ID
            data: Dictionary containing video metadata and transcript
            
        Returns:
            True if save was successful, False otherwise
        """
        transcript_text = data.get('transcript_text')
        title = data.get('title', 'N/A')
        publish_date = data.get('publish_date')
        length_seconds = data.get('length_seconds', 0)
        description = data.get('description', '')
        tags_list = data.get('tags', [])
        tags_str = ",".join(tags_list) if tags_list else None
        category_name = data.get('category_name', 'Unknown')

        # Count tokens if we have transcript text
        gemini_tokens, openai_tokens = self._count_tokens(transcript_text)

        try:
            if self.storage_type == 'file':
                if transcript_text:
                    filepath = os.path.join(self.output_dir, f"{video_id}.txt")
                    file_content = (
                        f"# Title: {title}\n"
                        f"# Published: {publish_date}\n"
                        f"# Length (s): {length_seconds}\n"
                        f"# Category: {category_name}\n"
                        f"# Tags: {tags_str if tags_str else 'None'}\n"
                        f"# Gemini Tokens: {gemini_tokens if gemini_tokens is not None else 'N/A'}\n"
                        f"# OpenAI Tokens: {openai_tokens if openai_tokens is not None else 'N/A'}\n"
                        f"# Description:\n# {'# '.join(description.splitlines())}\n\n"
                        f"{transcript_text}"
                    )
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(file_content)
                    logger.debug(f"Saved transcript to file: {filepath}")
                    return True
                else:
                    logger.warning(f"Skipping file save for {video_id} (no transcript)")
                    return False

            elif self.storage_type == 'sqlite' and self.conn:
                self.cursor.execute('''
                    INSERT OR REPLACE INTO transcripts
                    (video_id, title, publish_date, length_seconds, description, tags, 
                     category_name, transcript_text, gemini_tokens, openai_tokens)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (video_id, title, publish_date, length_seconds, description, tags_str,
                      category_name, transcript_text, gemini_tokens, openai_tokens))
                self.conn.commit()
                logger.debug(f"Saved transcript to database for video: {video_id}")
                return True
            else:
                logger.warning(f"Skipping DB save for {video_id}: Database not connected")
                return False

        except sqlite3.Error as e:
            logger.error(f"Database error saving data for {video_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error saving data for {video_id}: {e}")
            return False

    def download_channel_data(self, channel_url: str) -> int:
        """
        Download transcripts and metadata for all videos in a channel.
        
        Args:
            channel_url: URL of the YouTube channel
            
        Returns:
            Number of successfully processed videos
        """
        video_ids = self._get_channel_video_ids(channel_url)
        if not video_ids:
            logger.error("No video IDs found. Aborting.")
            return 0

        logger.info(f"Processing {len(video_ids)} videos...")
        successful = 0

        # Process IDs in batches
        for i in tqdm(range(0, len(video_ids), DEFAULT_BATCH_SIZE), desc="Processing Batches"):
            batch_ids = video_ids[i:i + DEFAULT_BATCH_SIZE]
            logger.info(f"Fetching metadata for batch {i//DEFAULT_BATCH_SIZE + 1}...")
            metadata_batch = self._fetch_metadata_batch(batch_ids)

            logger.info("Fetching transcripts and saving data for batch...")
            for video_id in batch_ids:
                video_metadata = metadata_batch.get(
                    video_id,
                    {
                        'title': 'N/A',
                        'publish_date': None,
                        'length_seconds': 0,
                        'description': '',
                        'tags': [],
                        'category_name': 'Unknown'
                    }
                )

                transcript = self._get_transcript(video_id)
                combined_data = {
                    **video_metadata,
                    'transcript_text': transcript
                }

                if self._save_data(video_id, combined_data):
                    successful += 1

        logger.info(f"Finished processing channel. Successfully processed {successful}/{len(video_ids)} videos")
        return successful


# count_tokens_in_directory needs slight adjustment for new file comment format
def count_tokens_in_directory(api_key: str, directory_path: str) -> Dict[str, int]:
    """
    Count tokens in transcript text from files, skipping metadata comments.
    
    Args:
        api_key: Gemini API key
        directory_path: Path to directory containing transcript files
        
    Returns:
        Dictionary mapping filenames to token counts
    """
    genai.configure(api_key=api_key)
    file_pattern = os.path.join(directory_path, "*.txt")
    text_files = glob.glob(file_pattern)
    token_counts = {}
    total_tokens = 0

    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content_lines = [line for line in f if not line.strip().startswith("#")]
                content = "".join(content_lines).strip()

            if not content:
                logger.warning(f"Skipping empty file: {os.path.basename(file_path)}")
                continue

            count_response = genai.count_tokens(
                model='models/gemini-1.5-flash-001',
                contents=content
            )
            token_count = count_response.total_tokens

            filename = os.path.basename(file_path)
            token_counts[filename] = token_count
            total_tokens += token_count
            logger.info(f"{filename}: {token_count} tokens")

        except Exception as e:
            logger.error(f"Error processing {file_path} for token count: {e}")

    logger.info(f"Total tokens across transcript files: {total_tokens}")
    return token_counts


def get_channel_name_from_url(url: str) -> str:
    """
    Extract channel name from YouTube channel URL.
    
    Handles various YouTube channel URL formats:
    - https://www.youtube.com/@channelname
    - https://www.youtube.com/c/channelname
    - https://youtube.com/@channelname (without www)
    - https://www.youtube.com/channel/UC... (channel ID format)
    
    Args:
        url: YouTube channel URL
        
    Returns:
        Channel name or ID (without @ symbol for @username format)
        
    Raises:
        ValueError: If URL is invalid or not a YouTube channel URL
    """
    # Remove any trailing slashes and convert to lowercase
    url = url.rstrip('/').lower()
    
    # Basic URL validation
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        # Handle @username format
        if '/@' in url:
            channel_name = url.split('/@')[-1].split('/')[0]
            if not channel_name:
                raise ValueError("Invalid channel name in URL")
            # Remove @ symbol if present at the start of the channel name
            return channel_name.lstrip('@')
            
        # Handle /c/ format
        elif '/c/' in url:
            channel_name = url.split('/c/')[-1].split('/')[0]
            if not channel_name:
                raise ValueError("Invalid channel name in URL")
            return channel_name
            
        # Handle channel ID format
        elif '/channel/' in url:
            channel_id = url.split('/channel/')[-1].split('/')[0]
            if not channel_id.startswith('UC'):
                raise ValueError("Invalid channel ID format")
            return channel_id
            
        else:
            raise ValueError("URL does not appear to be a valid YouTube channel URL")
            
    except Exception as e:
        raise ValueError(f"Failed to extract channel name from URL: {str(e)}")

def validate_channel_url(url: str) -> bool:
    """
    Validate if the URL is a potentially valid YouTube channel URL.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL appears to be a valid YouTube channel URL, False otherwise
    """
    try:
        get_channel_name_from_url(url)
        return True
    except ValueError:
        return False

def setup_argparse() -> argparse.ArgumentParser:
    """
    Set up command line argument parser.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Download and manage YouTube video transcripts and metadata",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "channel_url",
        nargs='?',  # Add this to make the argument optional
        default="https://www.youtube.com/@lexfridman",
        help="YouTube channel URL (e.g., https://www.youtube.com/@channelname, "
             "https://www.youtube.com/c/channelname, or https://www.youtube.com/channel/UC...)"
    )
    
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for storing transcripts (when using file storage). Default: {DEFAULT_OUTPUT_DIR}"
    )
    
    parser.add_argument(
        "--storage",
        choices=['file', 'sqlite'],
        default='sqlite',
        help="Storage type for transcripts and metadata"
    )
    
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION_CODE,
        help="Region code for category mapping (e.g., US, GB)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help="Set the logging level"
    )
    
    return parser

def main():
    """Main entry point for the YouTube Transcript Manager."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Validate channel URL
    if not validate_channel_url(args.channel_url):
        logger.error(f"Invalid YouTube channel URL: {args.channel_url}")
        logger.error("Please provide a valid YouTube channel URL in one of these formats:")
        logger.error("  - https://www.youtube.com/@channelname")
        logger.error("  - https://www.youtube.com/c/channelname")
        logger.error("  - https://www.youtube.com/channel/UC...")
        sys.exit(1)
    
    # Configure logging level
    logging.getLogger().setLevel(args.log_level)
    
    # Get API keys
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not youtube_api_key:
        logger.error("YouTube Data API key not found in .env file")
        logger.error("Please create a .env file with YOUTUBE_API_KEY=your_key_here")
        sys.exit(1)
    
    # Configure API clients
    if openai_api_key and openai:
        openai.api_key = openai_api_key
    
    # Set up storage paths
    if args.storage == 'sqlite':
        data_dir = Path(DEFAULT_DB_DIR)
        data_dir.mkdir(exist_ok=True, parents=True)  # Create parent directories if they don't exist
        channel_name = get_channel_name_from_url(args.channel_url)
        db_path = data_dir / f"{channel_name}.db"
    else:
        # Ensure the output directory exists
        os.makedirs(args.output_dir, exist_ok=True, parents=True)
        db_path = None
    
    try:
        # Initialize YouTube API service
        logger.info("Initializing YouTube Data API service...")
        youtube_service = build('youtube', 'v3', developerKey=youtube_api_key)
        logger.info("YouTube Data API service initialized successfully")
        
        # Initialize transcript manager
        transcript_manager = YouTubeTranscriptManager(
            youtube_service=youtube_service,
            output_dir=args.output_dir,
            storage_type=args.storage,
            db_path=str(db_path) if db_path else None,
            region_code=args.region
        )
        
        # Download transcripts and metadata
        logger.info(f"Starting download for channel: {args.channel_url}")
        successful = transcript_manager.download_channel_data(args.channel_url)
        
        logger.info(f"Download completed. Successfully processed {successful} videos")
        if args.storage == 'sqlite':
            logger.info(f"Data saved to database: {db_path}")
        else:
            logger.info(f"Data saved to directory: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if 'transcript_manager' in locals() and transcript_manager.storage_type == 'sqlite':
            transcript_manager.close_db()

if __name__ == "__main__":
    main()
    