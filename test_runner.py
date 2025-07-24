# test_runner.py

import os
from mat_utils import load_mat
from agents.vision_agent import VisionAgent
from agents.pdf_report import create_combined_pdf_report
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# === Config ===
DATA_DIR = "data"
OUTPUT_PDF = "output/all_reports_combined.pdf"
os.makedirs("output", exist_ok=True)


# Load environment variables from .env file
load_dotenv()

# Get API key for Gemini from env
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise EnvironmentError("GOOGLE_API_KEY not found in environment variables")



# Initialize Gemini LLM with API key
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key = google_api_key
)



def process_file(filepath, llm=None):
    filename = os.path.basename(filepath)
    name = os.path.splitext(filename)[0]

    print(f"[+] Processing: {filename}")
    img, label_map = load_mat(filepath)

    agent = VisionAgent(img, label_map, llm=llm)
    agent.analyze()
    agent.generate_visuals()

    insight = agent.generate_llm_insight()

    data = agent.get_report_data()
    return {
        "title": name,
        "results": data["results"],
        "images": data["images"],
        "captions": data["captions"],
        "llm_insight": insight
    }

# === Main Execution ===
def main():
    # List all .mat files in the data directory
    mat_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".mat")
    ]

    if not mat_files:
        print("No .mat files found in data/")
        return

    report_data = []
    for file_path in mat_files:
        try:
            report_data.append(process_file(file_path, llm=llm))
        except Exception as e:
            print(f"[!] Error processing {file_path}: {e}")

    print(f"[✓] Generating combined PDF report: {OUTPUT_PDF}")
    create_combined_pdf_report(OUTPUT_PDF, report_data)


if __name__ == "__main__":
    main()
