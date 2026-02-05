import os
import pandas as pd
import csv
from tqdm import tqdm  # For displaying progress bar (install with: pip install tqdm)

# ---------------------- Configuration Parameters ----------------------
# traffic_dir = "traffic/edge_base"  # Input folder path
# output_dir = "traffic/edge_merge"  # Output folder path
# traffic_dir = r"E:\SUMO_Outputs\2V_base\edge_base"  # Input folder path
# output_dir = r"E:\SUMO_Outputs\2V_base\edge_merge_small"  # Output folder path
traffic_dir = r"E:\SUMO_Outputs\edge_base_bal"  # Input folder path
output_dir = r"E:\SUMO_Outputs\predictor_training_data\18v\edge_merge_bal"  # Output folder path
# output_dir = r"E:\SUMO_Outputs\2V_base\edge_merge"  # Output folder path
id_column = "id"  # Column name used for grouping (i.e., the id column)
# ----------------------------------------------------------------------

# Create output folder
os.makedirs(output_dir, exist_ok=True)

# Get and sort all CSV files (by numbers in filenames)
def extract_num(file_name):
    try:
        return int(file_name.split("_")[-1].split(".")[0])
    except:
        return 0

csv_files = [f for f in os.listdir(traffic_dir) if f.startswith("edge_records") and f.endswith(".csv")]
csv_files.sort(key=extract_num)
total_files = len(csv_files)
print('csv_files',os.listdir(traffic_dir))
print('csv_files',csv_files)
# Record created ID files (to avoid duplicate headers)
created_ids = set()
blacklist_col = []
# blacklist_col = ["arrived","density","departed","entered","laneChangedFrom","laneChangedTo","laneDensity","left","occupancy","overlapTraveltime","sampledSeconds","speedRelative","teleported","timeLoss","traveltime","waitingTime"]

# Process files one by one (reduce memory usage)
for file_idx, file in enumerate(tqdm(csv_files, desc="Total Progress")):
    file_path = os.path.join(traffic_dir, file)
    try:
        # Read CSV file (use chunksize for chunked reading to further reduce memory pressure)
        # chunksize=10000 means reading 10,000 rows at a time; adjust based on cluster memory
        chunk_iter = pd.read_csv(file_path, chunksize=10000)
        
        for chunk in chunk_iter:
            # Check if id column exists
            if id_column not in chunk.columns:
                print(f"\nWarning: File {file} lacks {id_column} column, skipped")
                break
            # Remove blacklisted columns if they exist in the chunk
            chunk = chunk.drop(columns=[col for col in blacklist_col if col in chunk.columns], errors='ignore')
            
            # Group by ID and iterate through each ID's data
            for id_val, group in chunk.groupby(id_column, observed=True):
                # Handle special characters in ID (to avoid filename errors)
                safe_id = str(id_val).replace("/", "_").replace("\\", "_").replace(":", "_")
                output_path = os.path.join(output_dir, f"id_{safe_id}.csv")
                
                # Determine if file has been created (write header on first creation, append later)
                if id_val not in created_ids:
                    # Create file for the first time, write header and data
                    group.to_csv(output_path, index=False, mode="w", quoting=csv.QUOTE_MINIMAL)
                    created_ids.add(id_val)
                else:
                    # File already exists, append data only (no header)
                    group.to_csv(output_path, index=False, mode="a", header=False, quoting=csv.QUOTE_MINIMAL)
        
        print(f"\nCompleted {file_idx+1}/{total_files}: {file}")
    
    except Exception as e:
        print(f"\nError processing file {file}: {str(e)}")

print(f"\nAll files processed! Results saved to: {os.path.abspath(output_dir)}")