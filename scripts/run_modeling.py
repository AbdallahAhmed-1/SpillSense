# run_modeling.py

import argparse
import traceback
from state.state_utils import load_state, save_state
from agents.Modeler import Modeler


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2: Model Training")
    parser.add_argument(
        "--state",
        type=str,
        default="state/rca_state.joblib",
        help="Path to the saved RCA state file"
    )
    args = parser.parse_args()

    print(f"📦 Loading saved state from: {args.state}")
    try:
        state = load_state(args.state)
    except Exception as e:
        print(f"❌ Error loading state: {e}")
        traceback.print_exc()
        return

    modeler = Modeler()

    try:
        updated_state = modeler(state)
        print("✅ Modeling phase completed.")
    except Exception as e:
        print(f"❌ Error during modeling: {e}")
        traceback.print_exc()
        return

    # Save new state
    new_state_path = args.state.replace(".joblib", "_after_modeling.joblib")
    save_state(updated_state, new_state_path)
    print(f"💾 Updated state saved to: {new_state_path}")

if __name__ == "__main__":
    main()