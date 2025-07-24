# Agent that reads CSV/data files, validates data integrity,
# handles missing values, and outputs structured data to shared state.
# agents/loader.py
import pandas as pd
from typing import Optional, List
from state.state_utils import OilGasRCAState
from datetime import datetime
import os


class Loader:
    def __init__(self):
        self.name = "Loader"
    
    def __call__(self, state: OilGasRCAState) -> OilGasRCAState:
        """Make the class callable for LangGraph compatibility"""
        return self.run(state)
    
    def run(self, state: OilGasRCAState) -> OilGasRCAState:
        """
        Loads raw maintenance log data from file paths specified in the state.
        Supports CSV and JSON formats (extendable).
        Stores the raw DataFrame and metadata in the shared state.
        """
        # Get the paths list from state
        file_paths: Optional[List[str]] = state.get("paths")
        
        if not file_paths:
            state["errors_encountered"].append(
                {"agent": self.name, "error_message": "No file paths provided in state"}
            )
            state["agent_messages"].append({"from_agent": self.name, "message": "No valid data source path provided."})
            return state
        
        # Process each file path
        loaded_data = []
        processed_paths = []
        
        for file_path in file_paths:
            if not file_path or not os.path.exists(file_path):
                state["errors_encountered"].append(
                    {"agent": self.name, "error_message": f"Data source path invalid or not found: {file_path}"}
                )
                continue
            
            try:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".csv":
                    df = pd.read_csv(file_path)
                    source_type = "csv"
                elif ext == ".json":
                    df = pd.read_json(file_path)
                    source_type = "json"
                else:
                    state["warnings"].append(f"Unsupported file extension {ext}. Attempting CSV load.")
                    df = pd.read_csv(file_path)
                    source_type = "csv"
                
                # Store the loaded data
                loaded_data.append(df)
                processed_paths.append(file_path)
                
                state["agent_messages"].append({
                    "from_agent": self.name, 
                    "message": f"Successfully loaded data from {file_path}. Shape: {df.shape}"
                })
                
            except Exception as e:
                state["errors_encountered"].append(
                    {"agent": self.name, "error_message": f"Failed to load data from {file_path}: {str(e)}"}
                )
                continue
        
        # Update state with loaded data
        if loaded_data:
            # If multiple files, concatenate them (adjust logic as needed)
            if len(loaded_data) == 1:
                state["raw_data"] = loaded_data[0]
            else:
                # Concatenate multiple DataFrames
                state["raw_data"] = pd.concat(loaded_data, ignore_index=True)
            
            state["data_loaded"] = True
            state["data_loaded_at"] = datetime.now()
            state["data_source_type"] = source_type
            state["processed_paths"] = processed_paths
            
            # Clear downstream flags
            state["data_cleaned"] = False
            state["exploration_completed"] = False
            state["rca_completed"] = False
            state["report_generated"] = False
            
            state["agent_messages"].append({
                "from_agent": self.name, 
                "message": f"Loaded {len(loaded_data)} file(s) successfully. Total rows: {len(state['raw_data'])}"
            })
        else:
            state["agent_messages"].append({
                "from_agent": self.name, 
                "message": "No files could be loaded successfully."
            })
        
        return state