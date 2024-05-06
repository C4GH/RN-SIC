import csv
import re

# Path to your original log data file
log_file_path = 'losslog.txt'  # Update this with the actual path to your log file

# Path to the output CSV file for entries with model saved
csv_file_path = 'model_saved_entries_norm.csv'

# Regular expression to match the lines with model saved comment
pattern = re.compile(r'Epoch (\d+), Batch (\d+), Loss lowered to ([\d\.]+), model saved!')

# Reading from the log file and writing to a CSV file
with open(log_file_path, 'r') as file, open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Epoch', 'Batch', 'Loss'])  # Write headers to the CSV file

    # Read each line from the log file
    for line in file:
        match = pattern.search(line)
        if match:
            epoch, batch, loss = match.groups()
            writer.writerow([epoch, batch, loss])

print("CSV file has been created successfully for entries where the model was saved.")
