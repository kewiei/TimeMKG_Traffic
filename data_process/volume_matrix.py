import xml.etree.ElementTree as ET
import pandas as pd
import glob

# User's provided file path
#file_path = 'edge_records.xml'
file_paths = glob.glob(".\edge_records_*.xml")
print (file_paths)

for file_path in file_paths:
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Extracting specific edges' information
    edge_ids = [
        "1-70", "71-1", "1-2", "2-1", "10-2", "2-10", "1-8", "9-2", "3-1", "5-3", 
        "11-5", "5-6", "4-3", "2-4", "4-2", "4-6", "6-4", "6-12", "12-6"
    ]

    # Initialize a dictionary to store the data
    data = {edge_id: [] for edge_id in edge_ids}

    # Iterate through the XML file and extract data
    for interval in root.findall('interval'):
        time = interval.get('begin')
        interval_data = {edge_id: None for edge_id in edge_ids}
        
        for edge in interval.findall('edge'):
            edge_id = edge.get('id')
            if edge_id in edge_ids:
                try:
                    speed = float(edge.get('speed'))
                    density = float(edge.get('density'))
                    # Calculate average traffic volume using provided formula
                    average_traffic_volume = speed * 3.6 * density
                    interval_data[edge_id] = average_traffic_volume
                except:
                    pass

        # Add the data for this interval to the main data dictionary
        for edge_id in edge_ids:
            data[edge_id].append(interval_data[edge_id])

    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    df.index.name = 'Interval'

    # Save the DataFrame to a CSV file
    csv_file_path = 'volume_matrix_'+file_path[15:-4]+'.csv'
    df.to_csv(csv_file_path)
