import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
MODEL = "claude-sonnet-4-6"
REFINEMENT_ROUNDS = 3
PROFILE_PATH = "memory/user_profile.md"
OUTPUT_DIR = "output"
