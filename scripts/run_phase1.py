# scripts/run_phase1.py

from tools.graph_builder import build_graph
from state.state_utils import create_initial_state, save_state, OilGasRCAState
from state.session_data import session
from joblib import dump
import os
import traceback
from typing import cast


def run_csv_pipeline(input_path: str, report_path: str = "Reports"):
    """
    Executes Phase 1 of the RCA pipeline: loading, cleaning, exploring, and generating a PDF report.
    Saves state to disk for subsequent use by modeling or inference steps.
    """
    os.makedirs(report_path, exist_ok=True)

    print("📊 Starting Phase 1 pipeline...")
    graph = build_graph()

    initial_state = create_initial_state(
        modality="csv",
        data_path=input_path,
        pdf_report_path=report_path,
        target="Severity",
    )
    
    initial_state["analysis_params"] = {
        "test_size": 0.2,
        "random_state": 42,
        "scoring": "f1_macro",
        "target": "Severity",
    }

    try:
        final_state = graph.invoke(initial_state)

        print("\n✅ Phase 1 completed: Data loaded, cleaned, and explored.")

        # Save intermediate state
        intermediate_path = os.path.join(report_path, "state_after_eda.joblib")
        dump(final_state, intermediate_path)
        print(f"🧠 State saved to: {intermediate_path}")

        # Save main state
        state_typed = cast(OilGasRCAState, final_state)
        save_state(state_typed)
        session.state = state_typed

    except Exception as e:
        print(f"\n🚨 Error running Phase 1 pipeline: {str(e)}")
        traceback.print_exc()


def main():
    run_csv_pipeline("data/raw/csv/small_spill_incidents.csv", "Reports")


if __name__ == "__main__":
    main()
