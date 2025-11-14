import pandas as pd
import os
from tqdm import tqdm  # For displaying progress bar

# Folder path (adjust according to actual situation)
# folder_path = "traffic/edge_merge"
folder_path = r"E:\SUMO_Outputs\2V_base\edge_merge"

# Get list of all CSV files
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
total_files = len(csv_files)

if total_files == 0:
    print("No CSV files found, please check the folder path!")
else:
    print(f"Found {total_files} CSV files, starting merge...")
    all_data = []
    
    # Use tqdm to display progress bar, desc for description, total for total number of files
    for filename in tqdm(csv_files, desc="Merge Progress", unit="file"):
        file_path = os.path.join(folder_path, filename)
        # Read CSV (can add parameters like encoding if needed)
        df = pd.read_csv(file_path)
        # df = df[df['interval_begin'] <= 12000]
        # df = df[df['file_name'] == 'edge_records_11.xml']
        all_data.append(df)
    
    # Merge all data
    merged_df = pd.concat(all_data, ignore_index=True)
    
    # Save results
    output_file = os.path.join(folder_path, "merged_traffic.csv")
    merged_df.to_csv(output_file, index=False)
    
    print(f"\nMerge completed! Processed {total_files} files, total records: {len(merged_df)} rows")
    print(f"Results saved to: {output_file}")