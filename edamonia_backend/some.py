from edamonia_backend.logic.preprocessing.preprocess_data import process_data_frequency, process_data_embedded
from edamonia_backend.path_configs import INPUT_CORPUS_CSV_PATH, OUTPUT_CORPUS_CSV_PATH_FREQUENCY, \
    INPUT_QUESTION_CSV_PATH, OUTPUT_QUESTION_CSV_PATH_FREQUENCY

process_data_embedded(INPUT_CORPUS_CSV_PATH, OUTPUT_CORPUS_CSV_PATH_FREQUENCY)
process_data_embedded(INPUT_QUESTION_CSV_PATH, OUTPUT_QUESTION_CSV_PATH_FREQUENCY)
