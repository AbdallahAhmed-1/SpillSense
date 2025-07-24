# tools/image_graph_builder.py
"""Constructs the LangGraph workflow for hyperspectral `.mat` files.
Mirrors the CSV graph builder but plugs in MAT‑specific agents and passes
`modality="mat"` to the Reporter so it renders an HSI‑appropriate PDF.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

from state.state_utils import OilGasRCAState
from agents.image_loader import ImageLoader
from agents.image_cleaner import ImageCleaner
from agents.image_explorator import ImageExplorator
from agents.reporter import Reporter


def build_image_graph():
    """Return a compiled LangGraph ready to run on HSI cubes."""
    # 1. Load env & LLM
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not found in env vars")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
    )

    # 2. Instantiate modality‑specific agents
    loader = ImageLoader(label_map_file="data/labels/label_colors.txt")
    cleaner = ImageCleaner()
    explorator = ImageExplorator(llm=llm)
    # Use reporter.run as the callable, not the class instance
    reporter_agent = Reporter(llm=llm, mode="jpg")

    # 3. Build graph
    builder = StateGraph(OilGasRCAState)

    builder.add_node("loader", loader)
    builder.add_node("cleaner", cleaner)
    builder.add_node("explorator", explorator)
    builder.add_node("reporter", reporter_agent.run)

    builder.add_edge(START, "loader")
    builder.add_edge("loader", "cleaner")
    builder.add_edge("cleaner", "explorator")
    builder.add_edge("explorator", "reporter")
    builder.add_edge("reporter", END)

    return builder.compile()
