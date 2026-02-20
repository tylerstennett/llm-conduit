"""Load .env file for integration tests."""

from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root before any tests run.
_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_env_path, override=True)
