import json
import math


class JSONCleaner:
    def __init__(self):
        self.data = None
        self.entries_deleted = 0

    def load_and_clean_data(self, json_path):
        """Loads JSON data from a specified path and removes entries containing NaN values at a specific index."""
        try:
            with open(json_path, 'r') as file:
                data = json.load(file)

            original_length = len(data)
            # Filter out entries containing NaN values at the specified index
            self.data = [d for d in data if not (isinstance(d[1][0], float) and math.isnan(d[1][0]))]

            # Calculate how many entries were deleted
            self.entries_deleted = original_length - len(self.data)
            print(f"{self.entries_deleted} entries were deleted during cleaning.")
            print(f"Data loaded and cleaned from {json_path}")

        except FileNotFoundError:
            print(f"Error: The file {json_path} does not exist.")
        except json.JSONDecodeError:
            print("Error: Failed to decode JSON.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def save_cleaned_data(self, output_path):
        """Saves the cleaned data to a specified output JSON file."""
        if self.data is not None:
            with open(output_path, 'w') as file:
                json.dump(self.data, file, indent=4)
            print(f"Cleaned JSON has been saved to {output_path}")
        else:
            print("Error: No data to save. Please load and clean data first.")


# Define paths for input and output
input_path1 = r'C:\Users\afloresre\Documents\Cagh\Red\salida\embeddings 300\salida_entrenamiento.json'
output_path1 = r'C:\Users\afloresre\Documents\Cagh\Red\salida\embeddings 300\salida_entrenamiento_limpio.json'
input_path2 = r'C:\Users\afloresre\Documents\Cagh\Red\salida\embeddings 300\salida_prueba.json'
output_path2 = r'C:\Users\afloresre\Documents\Cagh\Red\salida\embeddings 300\salida_prueba_limpio.json'

# Create an instance of JSONCleaner
json_cleaner = JSONCleaner()

# Load, clean, and save the first file
json_cleaner.load_and_clean_data(input_path1)
json_cleaner.save_cleaned_data(output_path1)
print(f"Entries deleted from the first file: {json_cleaner.entries_deleted}")

# Load, clean, and save the second file
json_cleaner.load_and_clean_data(input_path2)
json_cleaner.save_cleaned_data(output_path2)
print(f"Entries deleted from the second file: {json_cleaner.entries_deleted}")
