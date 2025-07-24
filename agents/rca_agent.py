# agents/rca_agent.py

from langchain.agents import initialize_agent, AgentType
from tools.rca_tools import (
    train_model,
    predict_from_csv,
    get_model_metrics,
    get_confusion_matrix,
)
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Load environment variables (for Gemini API key)
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise EnvironmentError("GOOGLE_API_KEY not found in environment variables")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=google_api_key
)

# Register available RCA tools
TOOLS = [
    train_model,
    predict_from_csv,
    get_model_metrics,
    get_confusion_matrix,
]

# Initialize the agent
agent = initialize_agent(
    tools=TOOLS,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)
