# run_web_scraper.py
# Manual test script for WebScraperAgent

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from state.state_utils import OilGasRCAState
from agents.web_scraper import WebScraperAgent

# 1. Load environment variables
load_dotenv()
api_key = os.getenv("GNEWS_API_KEY")
if not api_key:
    raise EnvironmentError("GNEWS_API_KEY not found in environment variables")

# 2. Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")  # make sure this is set too if you plan to call Gemini
)

# 3. Prepare state with a sample natural-language query
# GNews free tier only provides recent articles (likely last 30-60 days)
state = OilGasRCAState()
state["web_search_input"] = (
   "Show me news articles about oil and gas from the last week")

# 4. Run the WebScraperAgent
agent = WebScraperAgent(llm=llm)
state = agent(state)

# 5. Display parsed terms and date range
print("Parsed terms:   ", state.get("web_search_terms"))
print("Date range:     ", state.get("web_search_start"), "→", state.get("web_search_end"))

# 6. Display fetched articles
results = state.get("web_search_results", None)

# If results is a dict (per-term), iterate .items()
if isinstance(results, dict):
    for term, articles in results.items():
        print(f"\n=== Results for '{term}' ({len(articles)} articles) ===")
        for a in articles:
            print(f"• {a['published_date']}: {a['title']} → {a['url']}")

# If results is a list (combined), iterate the list
elif isinstance(results, list):
    print(f"\n=== Combined Results ({len(results)} articles) ===")
    for idx, a in enumerate(results, 1):
        print(f"• {a['published_date']}: {a['title']} → {a['url']}")

else:
    print("No search results found.")
