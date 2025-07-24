import os, json, shutil, requests
from datetime import datetime, timedelta, date
from pathlib import Path
from dotenv import load_dotenv
from state.state_utils import OilGasRCAState

# choose the LLM you actually use elsewhere
from langchain_google_genai import ChatGoogleGenerativeAI   

load_dotenv()  # makes sure API keys are in env

# ---------- 1. Small helper --------------------------------------------------

def _clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json") :]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# ---------- 2. Thin public function -----------------------------------------

def scrape_news(query: str, output_dir: str = "static/reports") -> str:
    """
    Run the WebScraperAgent on `query` and return the PDF filename (basename only).
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    state = OilGasRCAState({"web_search_input": query})
    agent = WebScraperAgent(llm=llm)
    state = agent(state)

    pdf_path = state.get("pdf_report_path")
    if not pdf_path:
        raise RuntimeError("Scraper finished but produced no PDF.")

    os.makedirs(output_dir, exist_ok=True)
    dst = Path(output_dir) / Path(pdf_path).name
    shutil.move(pdf_path, dst)
    return dst.name  # backend /chat will move it into uploads/

# ---------- 3. The Agent -----------------------------------------------------

from langchain_core.language_models import BaseLanguageModel  # keep the import

class WebScraperAgent:
    """
    Parses natural-language query + optional dates, hits GNews API,
    and writes a PDF report.
    """

    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self.api_key = os.getenv("GNEWS_API_KEY")
        if not self.api_key:
            raise EnvironmentError("GNEWS_API_KEY not found in environment variables")

    # allow `agent(state)` style
    def __call__(self, state: OilGasRCAState) -> OilGasRCAState:
        return self.run(state)

    # ---------------------------------------------------------------------
    def run(self, state: OilGasRCAState) -> OilGasRCAState:
        # 1) grab query string
        query = state.get("web_search_input", "").strip()
        if not query:
            state.setdefault("warnings", []).append("WebScraper: no query provided.")
            return state

        # 2) ask LLM for terms + dates
        today = date.today()
        prompt = (
            "You are a search assistant.\n"
            "Given the user query below, output **only** a JSON object with:\n"
            '"terms": list of keyword strings,\n'
            '"start_date": optional YYYY-MM-DD,\n'
            '"end_date": optional YYYY-MM-DD.\n'
            "Apply the date-defaults described earlier.\n\n"
            f"User query:\n{query}"
        )
        raw = self.llm.invoke(prompt).content
        try:
            parsed = json.loads(_clean_json(raw))
            terms     = parsed.get("terms", [])
            start_iso = parsed.get("start_date")
            end_iso   = parsed.get("end_date")
        except Exception as e:
            state.setdefault("warnings", []).append(f"JSON parse failed: {e}")
            return state

        # 3) compute missing dates
        def to_d(s): return datetime.strptime(s, "%Y-%m-%d").date()
        start_dt = to_d(start_iso) if start_iso else None
        end_dt   = to_d(end_iso)   if end_iso   else None
        if start_dt and not end_dt: end_dt = start_dt + timedelta(days=7)
        if end_dt   and not start_dt: start_dt = end_dt - timedelta(days=7)
        if not start_dt and not end_dt:
            start_dt = today - timedelta(days=7)
            end_dt   = today

        # 4) GNews request
        combined = " ".join(f'"{t}"' for t in terms)
        url = (
            "https://gnews.io/api/v4/search"
            f"?q={requests.utils.quote(combined)}"
            f"&from={start_dt}&to={end_dt}"
            f"&lang=en&token={self.api_key}"
        )
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
        except Exception as e:
            state.setdefault("warnings", []).append(f"GNews error: {e}")
            articles = []

        # 5) filter + save results
        results = [
            {
                "title": a["title"],
                "summary": a["description"],
                "url": a["url"],
                "published_date": a["publishedAt"][:10],
            }
            for a in articles
            if any(t.lower() in (a["title"] + a["description"]).lower() for t in terms)
        ]

        # 6) build PDF right in output_dir to skip extra move
        
        pdf_name = f"web_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = Path("static/reports") / pdf_name
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet

            doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
            styles = getSampleStyleSheet()
            flow   = [
                Paragraph("Web Search Results", styles["Title"]),
                Spacer(1, 12),
                Paragraph(f"Query: {query}", styles["Normal"]),
                Paragraph(f"Date range: {start_dt} → {end_dt}", styles["Normal"]),
                Spacer(1, 12),
            ]
            for i, art in enumerate(results, 1):
                flow += [
                    Paragraph(f"<b>{i}. {art['title']}</b>", styles["Heading4"]),
                    Paragraph(f"Date: {art['published_date']}", styles["Normal"]),
                    Paragraph(art["summary"] or "No summary", styles["Normal"]),
                    Paragraph(art["url"], styles["Normal"]),
                    Spacer(1, 12),
                ]
            doc.build(flow)
        except ImportError:
            state.setdefault("warnings", []).append("ReportLab missing – no PDF")
            pdf_path = None

        # 7) push info into state
        state["web_search_results"]   = results
        state["pdf_report_path"]      = str(pdf_path) if pdf_path else ""
        state["web_search_completed"] = True
        return state
