import os
from pathlib import Path
from dotenv import load_dotenv

# Get project root directory
BASE_DIR = Path(__file__).resolve().parent

# Locate .env
ENV_PATH = BASE_DIR / ".env"

# Load environment variables
load_dotenv(ENV_PATH, override=True)

print("ENV file exists:", ENV_PATH.exists())

# Check environment variables
variables = [
    "OPENAI_API_KEY",
    "QDRANT_URL",
    "QDRANT_API_KEY",
    "QDRANT_COLLECTION"
]

for variable in variables:
    print(f"{variable} loaded:", bool(os.getenv(variable)))
    