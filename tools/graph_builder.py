# tools/graph_builder.py
# Constructs the LangGraph workflow, defines agent sequence/dependencies,
# handles state transitions and error flows.

from langgraph.graph import StateGraph, START, END
from state.state_utils import OilGasRCAState
from agents.loader import Loader
from agents.cleaner import Cleaner
from agents.explorator import Explorator
from agents.reporter import Reporter
from dotenv import load_dotenv
import os

# Import Gemini LLM from LangChain
from langchain_google_genai import ChatGoogleGenerativeAI


def build_graph():
    # Load environment variables from .env file
    load_dotenv()

    # Get API key for Gemini from env
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise EnvironmentError("GOOGLE_API_KEY not found in environment variables")

    # Initialize Gemini LLM with API key
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=google_api_key
    )

    # Optional: test the LLM
    try:
        response = llm.invoke("Hello, how are you?")
        print(f"LLM Response: {response}")
    except Exception as e:
        print(f"Error invoking LLM: {e}")

    # Instantiate agents
    loader = Loader()
    cleaner = Cleaner(llm=llm)
    explorator = Explorator(llm=llm)
    reporter = Reporter(llm=llm, mode="csv") 

    # Create graph with shared state
    builder = StateGraph(OilGasRCAState)

    # Add nodes
    builder.add_node("loader", loader)
    builder.add_node("cleaner", cleaner)
    builder.add_node("explorator", explorator)
    builder.add_node("reporter", reporter)  

    # Define Phase 1 sequence only
    builder.add_edge(START, "loader")
    builder.add_edge("loader", "cleaner")
    builder.add_edge("cleaner", "explorator")
    builder.add_edge("explorator", "reporter")  
    builder.add_edge("reporter", END)  # Stop here

    return builder.compile()
