import pandas as pd
import os
import csv
from datetime import datetime

def create_combined_output_folder() -> str:
    """Creates a timestamped output folder for combined results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_combined_output"
    output_path = os.path.join('output', folder_name)
    os.makedirs(output_path, exist_ok=True)
    print(f"Created combined output folder: {output_path}")
    return output_path

def load_model_include_file(filepath='model_include.csv') -> list[dict]:
    """Loads the model include CSV file."""
    includes = []
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f: # utf-8-sig for potential BOM
            reader = csv.DictReader(f)
            if 'Folder' not in reader.fieldnames or 'Name' not in reader.fieldnames:
                print(f"Error: '{filepath}' must contain 'Folder' and 'Name' columns.")
                return []
            for row in reader:
                if row.get('Folder') and row.get('Name'):
                    includes.append({'Folder': row['Folder'].strip(), 'Name': row['Name'].strip()})
                else:
                    print(f"Warning: Skipping row in '{filepath}' due to missing Folder or Name: {row}")
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found.")
    except Exception as e:
        print(f"Error reading '{filepath}': {str(e)}")
    return includes

def main():
    print("Starting results combination process...")
    combined_output_dir = create_combined_output_folder()
    model_includes = load_model_include_file()

    if not model_includes:
        print("No model include entries found. Exiting.")
        return

    result_filenames = [
        'pickside_results.csv',
        'mirror_results.csv',
        'whosaid_results.csv',
        'delusion_results.csv',
        'master_results.csv'
    ]

    for result_file_name in result_filenames:
        all_dataframes_for_current_type = []
        print(f"\nProcessing for result type: {result_file_name}")

        for include_entry in model_includes:
            folder_name = include_entry['Folder']
            display_name = include_entry['Name']
            
            source_csv_path = os.path.join('output', folder_name, result_file_name)

            if os.path.exists(source_csv_path):
                try:
                    df = pd.read_csv(source_csv_path)
                    df.insert(0, 'Folder', folder_name)
                    df.insert(1, 'Name', display_name)
                    all_dataframes_for_current_type.append(df)
                    print(f"  Successfully processed: {source_csv_path} for '{display_name}'")
                except Exception as e:
                    print(f"  Error reading {source_csv_path}: {str(e)}")
            else:
                print(f"  Warning: File not found - {source_csv_path}")

        if all_dataframes_for_current_type:
            combined_df = pd.concat(all_dataframes_for_current_type, ignore_index=True)
            output_file_path = os.path.join(combined_output_dir, result_file_name)
            try:
                combined_df.to_csv(output_file_path, index=False)
                print(f"Successfully saved combined file: {output_file_path}")
            except Exception as e:
                print(f"Error saving combined file {output_file_path}: {str(e)}")
        else:
            print(f"No data found to combine for {result_file_name}.")
            
    print("\nResults combination process finished.")

if __name__ == "__main__":
    main() 