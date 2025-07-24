from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from pathlib import Path
from tools.analysis_tools import generate_plot
import pandas as pd

from agents.llm_reasoner import LLMReasoner
from agents.web_scraper import scrape_news
from scripts.hsi_workflow import analyze_hsi_dataset
from scripts.predict_new_batch import predict_from_csv_file
from tools.image_tools import predict_image_spill

log = logging.getLogger(__name__)

# ------------------------------------------------------------
# Heuristics
# ------------------------------------------------------------
QUESTION_INDICATORS = {
    "why", "how", "what", "when", "analyze", "explain", "compare", "insight",
    "relationship", "correlation", "impact", "trend", "pattern", "significance",
    "tell me", "summarize", "overview", "assessment", "evaluate",
}


def _looks_like_question(text: str) -> bool:
    t = text.lower().strip()
    return t.endswith("?") or any(word in t for word in QUESTION_INDICATORS)


def _latest_csv(folder: Path) -> Optional[Path]:
    csvs = sorted(folder.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def make_figure(cmd: str, session_dir: Path, state) -> Path:
    """
    cmd: free text after 'figure' keyword
    returns: path to saved png
    """
    fig = generate_plot(cmd, state)  # your function should return a matplotlib fig
    out = session_dir / f"fig_{int(__import__('time').time())}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    return out


def _register_generated_file(session_dir: Path, filename: str, title: str, kind: str = "other"):
    """
    Optional artifact registration. Swallows all errors.
    """
    try:
        from state.artifact_store import register_artifact
        file_path = session_dir / filename
        if not file_path.exists():
            return

        name = filename.lower()
        if name.endswith(".csv"):
            mime = "text/csv"
        elif name.endswith(".pdf"):
            mime = "application/pdf"
        elif name.endswith((".png", ".jpg", ".jpeg")):
            mime = "image/png"
        elif name.endswith(".json"):
            mime = "application/json"
        else:
            mime = "application/octet-stream"

        register_artifact(session_dir, title, kind, file_path, mime)
        log.info("Registered artifact: %s -> %s", title, filename)
    except Exception as e:
        log.warning("Failed to register artifact %s: %s", filename, e)


# ------------------------------------------------------------
# Router
# ------------------------------------------------------------
def route_command(
    message: str,
    reasoner: Optional[LLMReasoner] = None,
    session_dir: Optional[Path] = None,
) -> Tuple[str, List[str]]:
    """
    Return (reply_text, [report_files]) for the given free-text command.
    `session_dir` is where this user's files live (uploads & generated).
    """
    if session_dir is None:
        return "❌ Internal error: session directory not provided.", []

    cmd = message.strip().lower()
    reports_dir = session_dir  # generated stuff stays per-session

    # 1) LLM reasoning for natural questions
    if reasoner and _looks_like_question(message):
        try:
            return reasoner.reason(message), []
        except Exception as e:
            log.exception("LLM Reasoning error")
            return (
                f"I encountered an error while analyzing your question: {e}. "
                "Please try rephrasing or providing more context.",
                [],
            )

    # 2) CSV prediction
    if "predict" in cmd and "csv" in cmd:
        # allow "predict csv myfile.csv"
        mfile = re.search(r"predict\s+csv\s+([^\s\"']+\.csv)", cmd)
        csv_path = (session_dir / mfile.group(1)) if mfile else _latest_csv(session_dir)
        if not csv_path or not csv_path.exists():
            return "❌ No CSV found in this session. Upload one first.", []

        try:
            # Force output INTO session_dir
            raw_out = predict_from_csv_file(str(csv_path), str(session_dir))
            raw_out_path = Path(raw_out)

            if raw_out_path.parent != session_dir:
                dst = session_dir / raw_out_path.name
                shutil.move(str(raw_out_path), dst)
                out_path = dst
            else:
                out_path = raw_out_path

            out_name = out_path.name

            # Register artifact
            _register_generated_file(
                session_dir,
                out_name,
                f"CSV Predictions - {datetime.now():%Y-%m-%d %H:%M}",
                "prediction",
            )

            # Insight for LLM
            if reasoner:
                try:
                    df = pd.read_csv(out_path)
                    reasoner.add_insight(
                        "csv_analysis",
                        {
                            "file": csv_path.name,
                            "total_records": int(df.shape[0]),
                            "predictions_made": int(df.shape[0]),
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                except Exception:
                    pass

            return f"✅ CSV prediction completed: **{out_name}**", [out_name]

        except Exception as e:
            log.exception("Error predicting CSV")
            return f"❌ Error predicting CSV: {e}", []

    # 3) Image analysis (explicit path)
    if "analyze" in cmd and "image" in cmd:
        m = re.search(r'["\']([^"\']+\.(?:jpg|jpeg|png|bmp))["\']', message, re.I)
        if not m:
            return "Please provide an image path in quotes (e.g., 'uploads/my_image.jpg').", []
        image_path_str = m.group(1)
        image_path = Path(image_path_str)
        if not image_path.is_absolute():
            image_path = session_dir / image_path

        if not image_path.exists():
            return f"❌ Image file not found: {image_path}", []

        try:
            result, confidence = predict_image_spill(str(image_path))

            result_filename = f"image_analysis_{datetime.now():%Y%m%d_%H%M%S}.json"
            result_path = session_dir / result_filename
            result_data = {
                "image_path": str(image_path),
                "result": result,
                "confidence": float(confidence),
                "timestamp": datetime.now().isoformat(),
                "analysis_type": "oil_spill_detection",
            }
            with result_path.open("w") as f:
                json.dump(result_data, f, indent=2)

            _register_generated_file(
                session_dir, result_filename, f"Image Analysis - {image_path.name}", "image_result"
            )

            if reasoner:
                reasoner.add_insight(
                    "image_analysis",
                    {
                        "file": image_path.name,
                        "has_spill": result == "Oil Spill Detected",
                        "confidence": float(confidence),
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            return f"📸 Image Analysis: {result} (Confidence: {confidence:.2%})", [result_filename]
        except Exception as e:
            log.exception("Error analyzing image")
            return f"❌ Error analyzing image: {e}", []

    # 4) HSI analysis
    if "analyze" in cmd and "hsi" in cmd:
        try:
            analyze_hsi_dataset()

            hsi_report = "hsi_analysis_report.pdf"
            hsi_path = session_dir / hsi_report

            reports_dir_root = Path("static/reports") / hsi_report
            if reports_dir_root.exists() and not hsi_path.exists():
                shutil.copy(reports_dir_root, hsi_path)

            if hsi_path.exists():
                _register_generated_file(
                    session_dir,
                    hsi_report,
                    f"HSI Analysis Report - {datetime.now():%Y-%m-%d %H:%M}",
                    "report",
                )

            if reasoner:
                reasoner.add_insight(
                    "hsi_analysis",
                    {
                        "analysis_type": "hyperspectral",
                        "timestamp": datetime.now().isoformat(),
                        "result_summary": "HSI analysis completed",
                    },
                )
            return "🌈 HSI analysis completed. Check the reports folder.", [hsi_report]
        except Exception as e:
            log.exception("HSI analysis error")
            return f"❌ Error in HSI analysis: {e}", []

    # 5) Web scraping
    if "scrape" in cmd or "news" in cmd:
        try:
            # everything after 'scrape news' (or 'news') becomes the query
            m = re.search(r"(scrape\s+news|news)(.*)", message, re.I)
            raw_query = (m.group(2) or "").strip() if m else message

            if not raw_query:
                return "Please provide a search query, e.g. `scrape news for ADNOC oil and gas`.", []

            # Your web scraper already handles date extraction from the query
            # and generates PDF directly, so we just call it
            pdf_filename = scrape_news(raw_query, output_dir=str(session_dir))
            
            # The scrape_news function returns just the filename, and the PDF 
            # is already saved in session_dir, so we just need to register it
            _register_generated_file(
                session_dir,
                pdf_filename,
                f"News Report - {raw_query}",
                "report"
            )

            # Extract some basic info for reasoner (optional)
            if reasoner:
                reasoner.add_insight(
                    "web_scraping",
                    {
                        "timestamp": datetime.now().isoformat(),
                        "query": raw_query,
                        "report_generated": pdf_filename,
                    },
                )

            return f"📰 Web search completed for **{raw_query}**. Generated PDF report.", [pdf_filename]
            
        except Exception as e:
            log.exception("News scraping error")
            return f"❌ Error scraping news: {e}", []

    # 6) Check spill (simple command for uploaded image path)
    if "check spill" in cmd:
        m = re.search(r"check spill\s+(.+)", message, re.I)
        if not m:
            return "Please provide an image path after 'check spill'.", []

        image_path = Path(m.group(1).strip())
        if not image_path.is_absolute():
            image_path = session_dir / image_path

        if not image_path.exists():
            return f"❌ Image file not found: {image_path}", []

        try:
            result, confidence = predict_image_spill(str(image_path))

            filename = f"spill_check_{datetime.now():%Y%m%d_%H%M%S}.json"
            out_path = session_dir / filename
            payload = {
                "image_path": str(image_path),
                "result": result,
                "confidence": float(confidence),
                "timestamp": datetime.now().isoformat(),
                "analysis_type": "spill_detection",
            }
            with out_path.open("w") as f:
                json.dump(payload, f, indent=2)

            _register_generated_file(
                session_dir, filename, f"Spill Check - {image_path.name}", "image_result"
            )

            if reasoner:
                reasoner.add_insight(
                    "image_analysis",
                    {
                        "file": image_path.name,
                        "has_spill": result == "Oil Spill Detected",
                        "confidence": float(confidence),
                        "timestamp": datetime.now().isoformat(),
                    },
                )

            return (
                f"🛢️ **Spill Detection Result for {image_path.name}:**\n\n"
                f"**{result}** (Confidence: {confidence:.2%})",
                [filename],
            )
        except Exception as e:
            log.exception("Error checking spill")
            return f"❌ Error checking spill in {image_path.name}: {e}", []
        
    # 7) Figures
    if "figure" in cmd or "plot" in cmd or "chart" in cmd:
        # everything after the keyword becomes the spec
        spec_match = re.search(r"(figure|plot|chart)\s+(.*)", message, re.I)
        spec = spec_match.group(2).strip() if spec_match else ""
        if not spec:
            return "Tell me what to plot, e.g. `figure correlation heatmap spill_severity`", []

        try:
            from state.state_utils import load_state  # or use cached state if you have one
            state = load_state("state/rca_state_after_modeling.joblib")  # or _cached_state()

            fig_path = make_figure(spec, session_dir, state)
            fname = fig_path.name

            _register_generated_file(session_dir, fname, f"Figure - {spec}", "figure")

            if reasoner:
                reasoner.add_insight("figure_request", {
                    "spec": spec,
                    "file": fname,
                    "timestamp": datetime.now().isoformat()
                })

            return f"🖼️ Figure generated: **{fname}**", [fname]
        except Exception as e:
            log.exception("Figure generation error")
            return f"❌ Could not generate figure: {e}", []    

    # 8) Help
    return (
        "I can help you with:\n"
        "- **predict csv [file.csv]** – Predict oil spill severity from CSV data\n"
        "- **analyze image \"path/to/image.jpg\"** – Detect oil spills in images\n"
        "- **check spill uploads/image.jpg** – Quick spill check for an uploaded image\n"
        "- **analyze hsi** – Run the hyperspectral (.mat) workflow and download the PDF report\n"
        "- **scrape news <keywords> <YYYY-MM-DD YYYY-MM-DD>** – Build a news JSON report on a topic\n"
        "- Ask me questions about the data and insights!",
        [],
    )
    

