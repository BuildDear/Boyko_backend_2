import os

from edamonia_backend.logic.preprocessing.embedded.preprocess_embedded import process_data
from edamonia_backend.path_configs import OUTPUT_CSV_PATH


def process_csv_pipeline(input_path=OUTPUT_CSV_PATH, output_path=None):
    """
    Generalized function to read data, process text, and save the result to a CSV file.
    :param input_path: Path to the input CSV file containing text data.
    :param output_path: Path to save the processed CSV file. Defaults to None,
                        in which case "_processed" is added to the input file name.
    :return: None. Saves the processed CSV to the specified output path.
    """
    if output_path is None:
        # If no output path is specified, use the same path with "_processed" appended to the file name
        output_path = os.path.splitext(input_path)[0] + "_processed.csv"

    try:
        # Process the data using the imported function
        process_data(input_path, output_path)
    except FileNotFoundError as e:
        # Handle the case where the input file is not found
        print(f"Error: {e}")
