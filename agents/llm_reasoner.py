from __future__ import annotations

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

log = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are SpillSense AI, an expert system for analyzing oil spills using multiple data sources.

You have access to insights from:
1. CSV data analysis - oil spill severity predictions, patterns, statistics
2. Hyperspectral imaging (HSI) - environmental impact, contamination spread
3. Image-based spill detection - visual confirmation of spills
4. Web-scraped news - recent incidents and reports

When answering questions:
- Be specific and cite data when available
- Look for patterns across different data sources
- Provide actionable insights
- If data is limited, acknowledge it
"""

# How many of each insight type we keep on disk (oldest dropped first)
MAX_ITEMS = {
    "csv_analysis": 10,
    "hsi_analysis": 10,
    "image_analysis": 20,
    "web_scraping": 5,
}


class LLMReasoner:
    """Agent that reasons across all analysis pipelines using Gemini."""

    def __init__(
        self,
        llm: Optional[ChatGoogleGenerativeAI] = None,
        system_prompt: Optional[str] = None,
        insights_path: str | Path = "state/pipeline_insights.json",
    ):
        self.llm: ChatGoogleGenerativeAI = llm or ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
        self.system_prompt: str = system_prompt or DEFAULT_SYSTEM_PROMPT

        self.insights_file = Path(insights_path)
        self.insights_file.parent.mkdir(parents=True, exist_ok=True)

        self.insights: Dict[str, List[Dict[str, Any]]] = self._load_insights()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_insight(self, insight_type: str, data: Dict[str, Any]) -> None:
        """Record a new insight from any pipeline (csv_analysis, hsi_analysis, ...)."""
        bucket = self.insights.setdefault(insight_type, [])

        # Ensure timestamp
        data.setdefault("timestamp", datetime.now().isoformat())
        bucket.append(data)

        # Trim bucket size if configured
        if insight_type in MAX_ITEMS:
            self.insights[insight_type] = bucket[-MAX_ITEMS[insight_type] :]

        self._save_insights()

    def reason(self, question: str) -> str:
        """Backward-compatible alias for external callers."""
        return self.reason_on_question(question)

    def reason_on_question(self, question: str) -> str:
        """Use the LLM to answer a natural-language question using stored insights."""
        context = self._format_insights_for_llm()

        messages = [
            SystemMessage(content=self.system_prompt + "\n\nCurrent accumulated insights are provided below."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
        ]

        try:
            response = self.llm.invoke(messages)
            # langchain_google_genai returns a BaseMessage; prefer .content if present
            return getattr(response, "content", str(response))
        except Exception as e:
            log.exception("LLM reasoning failed")
            return f"❌ LLM reasoning failed: {e}"

    def get_insights(self) -> Dict[str, Any]:
        """Return the in-memory insight store."""
        return self.insights

    def clear_insights(self) -> None:
        """Wipe all stored insights."""
        self.insights = {k: [] for k in MAX_ITEMS.keys()}
        self._save_insights()

    def update_llm(self, llm: ChatGoogleGenerativeAI) -> None:
        """Swap the underlying LLM instance."""
        self.llm = llm

    def set_system_prompt(self, prompt: str) -> None:
        """Override the system prompt."""
        self.system_prompt = prompt

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _format_insights_for_llm(self) -> str:
        """Turn the cached insights into a readable block for the LLM."""
        ctx_lines: List[str] = ["=== ACCUMULATED ANALYSIS INSIGHTS ===", ""]

        # CSV analyses
        csv_list = self.insights.get("csv_analysis", [])
        if csv_list:
            ctx_lines.append("## Oil Spill Data Analysis (CSV):")
            for analysis in csv_list[-3:]:
                ctx_lines.append(f"- {analysis.get('timestamp', 'N/A')}:")
                if 'file' in analysis:
                    ctx_lines.append(f"  File: {analysis['file']}")
                if 'total_records' in analysis:
                    ctx_lines.append(f"  Records: {analysis['total_records']}")
                if 'severity_distribution' in analysis:
                    ctx_lines.append(f"  Severity Distribution: {analysis['severity_distribution']}")
                if 'predictions_made' in analysis:
                    ctx_lines.append(f"  Predictions: {analysis['predictions_made']}")
            ctx_lines.append("")

        # HSI analyses
        hsi_list = self.insights.get("hsi_analysis", [])
        if hsi_list:
            ctx_lines.append("## Hyperspectral Imaging Analysis:")
            for analysis in hsi_list[-3:]:
                ctx_lines.append(f"- {analysis.get('timestamp', 'N/A')}:")
                if 'contamination_level' in analysis:
                    ctx_lines.append(f"  Contamination: {analysis['contamination_level']}")
                if 'affected_area' in analysis:
                    ctx_lines.append(f"  Area: {analysis['affected_area']}")
                if 'spectral_signatures' in analysis:
                    ctx_lines.append(f"  Signatures: {analysis['spectral_signatures']}")
                if 'environmental_impact' in analysis:
                    ctx_lines.append(f"  Impact: {analysis['environmental_impact']}")
            ctx_lines.append("")

        # Image analyses
        img_list = self.insights.get("image_analysis", [])
        if img_list:
            ctx_lines.append("## Visual Spill Detection:")
            total = len(img_list)
            spills = sum(1 for a in img_list if a.get('has_spill'))
            pct = (spills / total * 100) if total else 0
            ctx_lines.append(f"- Total images analyzed: {total}")
            ctx_lines.append(f"- Spills detected: {spills} ({pct:.1f}%)")
            ctx_lines.append("- Recent detections:")
            for analysis in img_list[-5:]:
                line = f"  {analysis.get('timestamp', 'N/A')}: "
                line += "SPILL DETECTED" if analysis.get('has_spill') else "No spill"
                if 'confidence' in analysis:
                    line += f" (confidence: {analysis['confidence']:.2f})"
                if 'image' in analysis:
                    line += f" [{analysis['image']}]"
                ctx_lines.append(line)
            ctx_lines.append("")

        # Web scraping
        web_list = self.insights.get("web_scraping", [])
        if web_list:
            ctx_lines.append("## Recent News & Reports:")
            for item in web_list[-2:]:
                ctx_lines.append(f"- {item.get('timestamp', 'N/A')}: Query: '{item.get('query', 'N/A')}'")
                results = item.get('results') or []
                if results:
                    ctx_lines.append(f"  Found {len(results)} articles")
                    for article in results[:3]:
                        if isinstance(article, dict):
                            ctx_lines.append(f"  • {article.get('title', 'No title')}")
                            if 'summary' in article:
                                ctx_lines.append(f"    {article['summary'][:100]}...")
                if 'pdf_report' in item:
                    ctx_lines.append(f"  Report: {item['pdf_report']}")
            ctx_lines.append("")

        return "\n".join(ctx_lines)

    def _load_insights(self) -> Dict[str, Any]:
        """Load insight cache from disk, or create a fresh structure."""
        if self.insights_file.exists():
            try:
                with open(self.insights_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Ensure all expected keys exist
                    for k in MAX_ITEMS.keys():
                        data.setdefault(k, [])
                    return data
            except Exception:
                log.exception("Failed to load insights JSON; starting fresh.")
        return {k: [] for k in MAX_ITEMS.keys()}

    def _save_insights(self) -> None:
        """Persist insight cache to disk."""
        try:
            with open(self.insights_file, "w", encoding="utf-8") as f:
                json.dump(self.insights, f, indent=2)
        except Exception:
            log.exception("Failed to save insights JSON")
