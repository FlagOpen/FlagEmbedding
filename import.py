import csv

# Script for taking columns from a csv
def retrieve_lines(index_file, data_file, output_file):
    token_lines = {}
    
    # Read cleaned indices and their corresponding lines into a dictionary
    with open(index_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:  # Ensure there are tokens in the row
                token = row[0].strip()  # Assuming token is in the first column
                token_lines[token] = None  # Initialize with None or an empty list
    
    # Read data file and match tokens to retrieve lines
    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line in reader:
            if line:  # Ensure there are values in the line
                token = line[0].strip()  # Assuming token is in the first column
                if token in token_lines:
                    if token_lines[token] is None:
                        token_lines[token] = []  # Initialize as a list
                    token_lines[token].append(line)
    
    # Write retrieved lines to output file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for token, lines in token_lines.items():
            for line in lines:
                writer.writerow(line)

# Example usage:
if __name__ == "__main__":
    index_file = "cleanedCWEs.csv"    # Replace with your cleaned index file path
    data_file = "cwe_data.csv"  # Replace with your data file ('y') path
    output_file = "relevantCWE.csv"  # Replace with desired output file path
    
    retrieve_lines(index_file, data_file, output_file)
    print(f"Output written to {output_file}")
