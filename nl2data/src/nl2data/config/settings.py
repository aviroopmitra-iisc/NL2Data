"""Application settings using Pydantic Settings."""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


def find_and_load_env_file():
    """Find and load .env file in current directory or parent directories."""
    # Try multiple starting points
    search_paths = [
        Path.cwd(),  # Current working directory
        Path(__file__).parent.parent.parent.parent.parent,  # Project root
    ]
    
    for start_path in search_paths:
        current = start_path.resolve()
        # Check current directory and up to 3 levels up
        for _ in range(4):
            env_path = current / ".env"
            if env_path.exists():
                # Load the .env file into environment variables
                load_dotenv(env_path, override=False)
                return str(env_path)
            parent = current.parent
            if parent == current:  # Reached root
                break
            current = parent
    return None


# Load .env file before Settings class is defined
find_and_load_env_file()


class Settings(BaseSettings):
    """Application configuration settings."""

    # LLM Configuration (supports Gemini)
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    # Legacy OpenAI support (optional)
    openai_api_key: Optional[str] = None
    model_name: Optional[str] = None
    # Local LLM support (OpenAI-compatible API)
    llm_url: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.2
    
    # LLM Timeout Configuration (in seconds)
    llm_timeout: float = 1800.0  # 30 minutes default for long-running requests
    llm_max_retries: int = 3  # Maximum retry attempts for failed requests
    llm_retry_delay: float = 5.0  # Initial retry delay in seconds (exponential backoff)

    # Generation Configuration
    seed: int = 7
    output_dir: Path = Path("output")
    chunk_rows: int = 1_000_000

    # Logging Configuration
    log_level: str = "INFO"
    log_file: Optional[Path] = None

    model_config = SettingsConfigDict(
        env_file=None,  # We load it manually with dotenv
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def __init__(self, **kwargs):
        """Initialize settings and create output directory if needed."""
        # Ensure .env is loaded (in case this is called from a different directory)
        find_and_load_env_file()
        super().__init__(**kwargs)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.log_file:
            self.log_file = Path(self.log_file)
            self.log_file.parent.mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

