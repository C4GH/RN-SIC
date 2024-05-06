import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
data = pd.read_csv('training_loss.csv')

# Display the first few rows to inspect the data
print(data.head())

# Check for any missing values in the DataFrame
if data.isnull().any().any():
    print("Warning: Missing data detected! Please check your CSV file.")
else:
    print("No missing data detected.")

# Plotting a separate graph for each epoch
for epoch in sorted(data['Epoch'].unique()):
    epoch_data = data[data['Epoch'] == epoch]
    plt.figure(figsize=(10, 5))  # New figure for each epoch
    plt.plot(epoch_data['Batch'], epoch_data['Loss'], marker='o', linestyle='-')
    plt.title(f'Training Loss for Epoch {epoch}')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

