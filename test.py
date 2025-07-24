from tools.rca_tools import predict_from_csv

# Try it on a sample file
new_file = predict_from_csv("data/raw/csv/simulated_test_input.csv")

if new_file:
    print(f"✅ Saved: static/reports/{new_file}")
else:
    print("⚠️ Already predicted, skipping.")
