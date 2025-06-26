"""Input validation utilities."""

import re
from typing import Optional
from urllib.parse import urlparse


def is_valid_youtube_url(url: str) -> bool:
    """Validate YouTube video or channel URL."""
    youtube_patterns = [
        r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'https?://(?:www\.)?youtube\.com/channel/[\w-]+',
        r'https?://(?:www\.)?youtube\.com/@[\w-]+',
        r'https?://youtu\.be/[\w-]+',
    ]
    
    return any(re.match(pattern, url) for pattern in youtube_patterns)


def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL."""
    video_id_patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in video_id_patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def extract_channel_id(url: str) -> Optional[str]:
    """Extract channel ID or handle from YouTube URL."""
    channel_patterns = [
        r'youtube\.com/channel/([a-zA-Z0-9_-]+)',
        r'youtube\.com/@([a-zA-Z0-9_-]+)',
    ]
    
    for pattern in channel_patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def validate_token_limit(token_count: int, max_tokens: int = 1000000) -> bool:
    """Validate that token count is within acceptable limits."""
    return 0 <= token_count <= max_tokens


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    # Remove or replace unsafe characters
    safe_chars = re.sub(r'[<>:"/\\|?*]', '_', filename)
    return safe_chars.strip()[:255]  # Limit length 