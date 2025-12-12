import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure we load the .env that lives inside the healthbackend/ folder
_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# API_KEY = os.getenv("API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
