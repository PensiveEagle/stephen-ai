# ----- Import packages ----- #
from dotenv import load_dotenv
import os

# ----- Get env variables ----- #
load_dotenv()

todoist_api_key = os.getenv( "TODOIST_API_KEY" )
gemini_api_key = os.getenv( "GEMINI_API_KEY" )
