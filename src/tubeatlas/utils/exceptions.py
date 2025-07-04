"""Custom exceptions for TubeAtlas."""


class TubeAtlasException(Exception):
    """Base exception for TubeAtlas."""

    pass


class YouTubeAPIException(TubeAtlasException):
    """Exception for YouTube API related errors."""

    pass


class QuotaExceededError(YouTubeAPIException):
    """Exception raised when YouTube API quota is exceeded."""

    pass


class TransientAPIError(YouTubeAPIException):
    """Exception for transient API errors that can be retried."""

    pass


class TranscriptNotAvailableException(TubeAtlasException):
    """Exception when transcript is not available for a video."""

    pass


class TokenLimitExceededException(TubeAtlasException):
    """Exception when token limit is exceeded."""

    pass


class KnowledgeGraphGenerationException(TubeAtlasException):
    """Exception during knowledge graph generation."""

    pass


class DatabaseException(TubeAtlasException):
    """Exception for database-related errors."""

    pass


class ValidationException(TubeAtlasException):
    """Exception for input validation errors."""

    pass


class ConfigurationException(TubeAtlasException):
    """Exception for configuration-related errors."""

    pass


class TranscriptDownloadError(TubeAtlasException):
    """Exception raised when transcript download fails after retries."""

    pass
