import os

from dotenv import load_dotenv

load_dotenv()

WORKING_MEMORY_TOKEN_THRESHOLD = int(
    os.environ.get("WORKING_MEMORY_TOKEN_THRESHOLD", "1500")
)

ENVIRONMENT = os.environ.get("ENVIRONMENT", "local")
