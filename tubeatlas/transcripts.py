# Standard library imports
import glob
import os
import re
import sqlite3
from functools import partial
from multiprocessing import Pool
import datetime # For handling dates
import json # For potentially storing tags as JSON if preferred

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
    from dotenv import load_dotenv # For API Key
except ImportError:
    load_dotenv = lambda: None
try:
    from googleapiclient.discovery import build # For YouTube Data API
    from googleapiclient.errors import HttpError # For API errors
except ImportError:
    build = None
    HttpError = Exception
try:
    import isodate  # For ISO 8601 duration parsing
except ImportError:
    isodate = None


class YouTubeTranscriptManager:
    """Manages downloading YouTube transcripts and metadata using APIs."""
    # Add youtube_service parameter to __init__
    def __init__(self, youtube_service, output_dir="transcripts", storage_type='file', db_path='transcripts.db', region_code='US'):
        """
        Initialize the manager and fetch category mapping.

        Args:
            youtube_service: Initialized YouTube Data API service object.
            output_dir (str): Directory for file storage.
            storage_type (str): 'file' or 'sqlite'.
            db_path (str): Path for SQLite DB.
            region_code (str): Region for fetching category names (e.g., 'US', 'GB').
        """
        if not youtube_service:
             raise ValueError("YouTube Data API service object is required.")
        self.youtube_service = youtube_service
        self.output_dir = output_dir
        self.storage_type = storage_type.lower()
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.category_mapping = {} # Initialize category mapping

        if self.storage_type not in ['file', 'sqlite']:
            raise ValueError("storage_type must be either 'file' or 'sqlite'")

        if self.storage_type == 'file':
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        elif self.storage_type == 'sqlite':
            self._initialize_db()

        # Fetch category mapping immediately after initialization
        self._fetch_category_mapping(region_code)

    def _fetch_category_mapping(self, region_code='US'):
        """Fetch mapping from category ID to category name."""
        print(f"Fetching YouTube category mapping for region: {region_code}...")
        mapping = {}
        try:
            request = self.youtube_service.videoCategories().list(
                part="snippet",
                regionCode=region_code
            )
            response = request.execute()
            for item in response.get('items', []):
                mapping[item['id']] = item['snippet']['title']
            self.category_mapping = mapping
            print(f"Successfully fetched {len(mapping)} categories.")
        except HttpError as e:
            print(f"API error fetching categories: {e}. Proceeding without category names.")
            self.category_mapping = {} # Ensure it's empty on error
        except Exception as e:
             print(f"Unexpected error fetching categories: {e}. Proceeding without category names.")
             self.category_mapping = {}

    def _initialize_db(self):
        """Initialize SQLite DB and ensure table includes all metadata columns."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            # Ensure table schema includes title, publish_date, length_seconds, description, tags, category_name
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
                    download_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            self.conn.commit()
            print(f"Connected to SQLite database: {self.db_path}")
        except sqlite3.Error as e:
            print(f"Error connecting to or initializing SQLite database: {e}")
            self.conn = None
            self.cursor = None

    def close_db(self):
        """Close the SQLite database connection if it's open."""
        if self.conn:
            self.conn.close()
            print("SQLite database connection closed.")
            self.conn = None
            self.cursor = None

    def __del__(self):
        """Ensure database connection is closed when the object is destroyed."""
        self.close_db()

    def _get_channel_video_ids(self, channel_url):
        """Get video IDs from a YouTube channel URL using scrapetube."""
        print(f"Getting video IDs from channel URL {channel_url} using scrapetube...")
        try:
            videos_raw = scrapetube.get_channel(channel_url=channel_url)
            video_ids = [video['videoId'] for video in videos_raw]
            print(f"Found {len(video_ids)} video IDs.")
            return video_ids
        except Exception as e:
             print(f"Error getting video IDs using scrapetube for URL {channel_url}: {e}")
             return []

    def _parse_duration(self, duration_str):
        """Parses ISO 8601 duration string to total seconds."""
        if not duration_str:
            return 0
        # If isodate library is unavailable, apply simple regex fallback
        if isodate is None:
            import re
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
        except Exception:
            return 0

    def _fetch_metadata_batch(self, video_ids_batch):
        """Fetch metadata (snippet, contentDetails) including tags and category."""
        metadata = {}
        if not self.youtube_service:
             print("Error: YouTube service not initialized.")
             return metadata

        try:
            # Request both snippet and contentDetails parts
            request = self.youtube_service.videos().list(
                part="snippet,contentDetails", # <-- Added contentDetails
                id=",".join(video_ids_batch)
            )
            response = request.execute()

            for item in response.get('items', []):
                video_id = item['id']
                snippet = item.get('snippet', {})
                content_details = item.get('contentDetails', {}) # <-- Get contentDetails

                title = snippet.get('title', 'N/A')
                description = snippet.get('description', '') # <-- Extract description

                # Parse publish date
                published_at_str = snippet.get('publishedAt')
                publish_date = None
                if published_at_str:
                    try:
                       publish_date = datetime.datetime.fromisoformat(published_at_str.replace('Z', '+00:00')).strftime('%Y-%m-%d')
                    except ValueError:
                        print(f"Warning: Could not parse date '{published_at_str}' for video {video_id}")

                # Parse duration from contentDetails
                duration_iso = content_details.get('duration') # <-- Extract duration string
                length_seconds = self._parse_duration(duration_iso) # <-- Parse to seconds

                # --- Get Tags and Category ---
                tags_list = snippet.get('tags', []) # Get tags list (or empty list)
                category_id = snippet.get('categoryId') # Get category ID string
                # Look up category name from stored mapping
                category_name = self.category_mapping.get(category_id, 'Unknown') if category_id else 'Unknown'
                # -----------------------------

                metadata[video_id] = {
                    'title': title,
                    'publish_date': publish_date,
                    'description': description, # <-- Add description
                    'length_seconds': length_seconds, # <-- Add parsed length
                    'tags': tags_list, # Store the list for now
                    'category_name': category_name # Store the looked-up name
                }
        except HttpError as e:
            print(f"An API error occurred during metadata fetch: {e}")
            if e.resp.status == 403:
                print("Quota possibly exceeded. Check Google Cloud Console.")
        except Exception as e:
            print(f"An unexpected error occurred during metadata fetch: {e}")

        return metadata

    def _get_transcript(self, video_id):
        """Fetch transcript text directly without timestamps for a single video ID."""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            # Directly join the 'text' parts
            text = "\n".join(entry['text'] for entry in transcript_list)
            return text.strip() # Return cleaned, timestamp-free text
        except Exception as e:
            print(f"Could not get transcript for {video_id}: {e}")
            return None

    def _save_data(self, video_id, data):
        """Saves the combined transcript and all fetched metadata, including tags/category."""
        transcript_text = data.get('transcript_text')
        title = data.get('title', 'N/A')
        publish_date = data.get('publish_date')
        length_seconds = data.get('length_seconds', 0) # <-- Get length
        description = data.get('description', '')   # <-- Get description
        # Join tags list into a comma-separated string for DB storage
        tags_list = data.get('tags', [])
        tags_str = ",".join(tags_list) if tags_list else None # Store as comma-separated or NULL
        category_name = data.get('category_name', 'Unknown')

        try:
            if self.storage_type == 'file':
                if transcript_text:
                    filepath = os.path.join(self.output_dir, f"{video_id}.txt")
                    # Include all metadata as comments
                    file_content = (
                        f"# Title: {title}\n"
                        f"# Published: {publish_date}\n"
                        f"# Length (s): {length_seconds}\n"
                        f"# Category: {category_name}\n"
                        f"# Tags: {tags_str if tags_str else 'None'}\n"
                        f"# Description:\n# {'# '.join(description.splitlines())}\n\n" # Indent description lines
                        f"{transcript_text}"
                    )
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(file_content)
                    return True
                else:
                    print(f"Skipping file save for {video_id} (no transcript).")
                    return False

            elif self.storage_type == 'sqlite' and self.conn:
                # Update INSERT statement to include length_seconds, description, tags, and category_name
                self.cursor.execute('''
                    INSERT OR REPLACE INTO transcripts
                    (video_id, title, publish_date, length_seconds, description, tags, category_name, transcript_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (video_id, title, publish_date, length_seconds, description, tags_str, category_name, transcript_text))
                self.conn.commit()
                return True
            elif self.storage_type == 'sqlite':
                print(f"Skipping DB save for {video_id}: Database not connected.")
                return False
            else:
                return False

        except sqlite3.Error as e:
             print(f"Error inserting data for {video_id} into database: {e}")
             return False
        except Exception as e:
             print(f"Error saving file for {video_id}: {e}")
             return False

    def download_channel_data(self, channel_url):
        """Download transcripts and metadata for a channel URL."""
        video_ids = self._get_channel_video_ids(channel_url)
        if not video_ids:
            print("No video IDs found. Aborting.")
            return 0

        print(f"Processing {len(video_ids)} videos...")
        successful = 0
        batch_size = 50 # Max IDs per videos.list call

        # Process IDs in batches for metadata fetching efficiency
        for i in tqdm(range(0, len(video_ids), batch_size), desc="Processing Batches"):
            batch_ids = video_ids[i:i + batch_size]
            print(f"\nFetching metadata for batch {i//batch_size + 1}...")
            metadata_batch = self._fetch_metadata_batch(batch_ids) # Fetches desc/length now

            print("Fetching transcripts and saving data for batch...")
            for video_id in batch_ids:
                # Get potentially richer metadata for this specific video
                video_metadata = metadata_batch.get(
                    video_id,
                    {'title': 'N/A', 'publish_date': None, 'length_seconds': 0, 'description': '', 'tags': [], 'category_name': 'Unknown'} # Default values
                )

                # Get transcript individually
                transcript = self._get_transcript(video_id)

                # Combine data
                combined_data = {
                    **video_metadata, # Includes title, publish_date, length_seconds, description, tags, category_name
                    'transcript_text': transcript
                }

                # Save the combined data (including length/description)
                if self._save_data(video_id, combined_data):
                    successful += 1
                # time.sleep(0.1) # Optional delay

        print(f"\nFinished processing.")
        if self.storage_type == 'file':
            print(f"Saved data for {successful}/{len(video_ids)} videos to {self.output_dir}")
        elif self.storage_type == 'sqlite':
            print(f"Saved data for {successful}/{len(video_ids)} videos to database {self.db_path}")
        return successful


# count_tokens_in_directory needs slight adjustment for new file comment format
def count_tokens_in_directory(api_key, directory_path):
    """
    Count tokens in transcript text from files, skipping metadata comments.
    """
    genai.configure(api_key=api_key)
    file_pattern = os.path.join(directory_path, "*.txt")
    text_files = glob.glob(file_pattern)
    token_counts = {}
    total_tokens = 0

    for file_path in text_files:
        try:
            # Read file content, skipping ALL metadata comment lines
            with open(file_path, 'r', encoding='utf-8') as f:
                 # More robust skipping of multi-line description comments too
                 content_lines = [line for line in f if not line.strip().startswith("#")]
                 content = "".join(content_lines).strip()

            if not content: # Skip if only comments were present
                print(f"Skipping token count for empty/comment-only file: {os.path.basename(file_path)}")
                continue

            count_response = genai.count_tokens(model='models/gemini-1.5-flash-001', contents=content)
            token_count = count_response.total_tokens

            filename = os.path.basename(file_path)
            token_counts[filename] = token_count
            total_tokens += token_count
            print(f"{filename}: {token_count} tokens")

        except Exception as e:
            print(f"Error processing {file_path} for token count: {e}")

    print(f"\nTotal tokens across transcript files: {total_tokens}")
    return token_counts


# KnowledgeGraphBuilder class definition - assumed present but separate
# class KnowledgeGraphBuilder:
#    ...


def get_channel_name_from_url(url: str):
    channel_name = url.split("@")[-1]
    return channel_name

if __name__ == "__main__":
    # Load environment variables (for API key)
    load_dotenv()
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

    if not YOUTUBE_API_KEY:
        print("Error: YouTube Data API key not found in .env file under YOUTUBE_API_KEY")
        print("Please create a .env file in the same directory with the line:")
        print("YOUTUBE_API_KEY=YOUR_API_KEY_HERE")
        exit() # Exit if no API key

    # --- Configuration ---
    channel_url = "https://www.youtube.com/@AndrejKarpathy"
    transcript_dir = "transcripts" # Used only if storage = 'file'
    # gemini_api_key = os.getenv("GEMINI_API_KEY") # Example if you add Gemini key to .env

    # Choose storage type: 'file' or 'sqlite'
    storage = 'sqlite' # Options: 'file', 'sqlite'
    # Organize database files under the data directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    # Use a per-channel SQLite DB in data/
    db_file = os.path.join(data_dir, f"{get_channel_name_from_url(channel_url)}.db")
    region_code_for_categories = 'US' # Define region for category names

    transcript_manager = None
    youtube_service = None # Variable for the API service

    try:
        # --- Initialize YouTube Data API Service ---
        print("Initializing YouTube Data API service...")
        try:
            youtube_service = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
            print("YouTube Data API service initialized.")
        except Exception as api_init_error:
             print(f"Error initializing YouTube Data API service: {api_init_error}")
             exit() # Exit if API service fails to initialize


        # --- Workflow ---

        # 1. Initialize Manager (pass the youtube_service)
        if storage == 'file':
            transcript_manager = YouTubeTranscriptManager(youtube_service, output_dir=transcript_dir, storage_type='file', region_code=region_code_for_categories)
        elif storage == 'sqlite':
            transcript_manager = YouTubeTranscriptManager(youtube_service, storage_type='sqlite', db_path=db_file, region_code=region_code_for_categories)
        else:
            raise ValueError("Invalid storage type selected.")

        # 2. Download Transcripts & Metadata using the combined approach
        print(f"--- Downloading Transcripts & Metadata ---")
        transcript_manager.download_channel_data(channel_url)

        print(f"\n--- Workflow Complete ---")

    except Exception as e:
        print(f"An error occurred during the workflow: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure DB connection is closed if it was opened
        if transcript_manager and transcript_manager.storage_type == 'sqlite':
            transcript_manager.close_db()
        # No need to explicitly close the youtube_service object
    