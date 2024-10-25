import csv
import json

def csv_to_json(csv_file, json_file):
    data = []

    # Read CSV file and extract second and third values from each row
    with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)  # Skip header row if present
        
        for row in reader:
            if len(row) >= 3:  # Ensure at least 3 values in the row
                row_data = {
                    "Title": row[1],
                    "Description": row[2]
                }
                data.append(row_data)
    
    # Write data to JSON file
    with open(json_file, 'w', encoding='utf-8') as jsonfile:
        json.dump(data, jsonfile, indent=4)

# Example usage:
if __name__ == "__main__":
    csv_file = "relevantCWE.csv"    # Replace with your CSV file path
    json_file = "CWEs.json"  # Replace with desired output JSON file path
    
    csv_to_json(csv_file, json_file)
    print(f"Conversion from {csv_file} to {json_file} complete.")
