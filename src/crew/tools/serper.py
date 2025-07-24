from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

SerperTool = SerperDevTool()