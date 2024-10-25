import csv

def read_csv_to_list(file_path, column_index=0):
    data_list = []
    with open(file_path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row) > column_index:  # Ensure the column exists
                data_list.append(row[column_index])  # Append the value from the specified column
    return data_list