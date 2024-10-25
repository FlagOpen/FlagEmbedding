import csv

# Replace with your input and output file paths
input_file = 'relevantCWE.csv'
output_file = 'CWEdescriptions.csv'

def read_csv_to_array(file_path):
    data_array = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            data_array.append(row)  # Append each row as a list
    return data_array

with open(input_file, mode='r') as infile, open(output_file, mode='w', newline='') as outfile:
    csv_reader = csv.reader(infile)
    csv_writer = csv.writer(outfile)
    
    for row in csv_reader:
        if len(row) >= 3:  # Ensure there are at least 3 columns
            csv_writer.writerow([row[2]])  # Write the third column
    


print("Let's hope this worked!")