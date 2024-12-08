import re
import string
import ftfy
import emoji
import pandas as pd
import pymorphy2
from tqdm import tqdm

from edamonia_backend.logic.preprocessing.stop_words import century_words, additional_replacements, number_words, \
    ukrainian_stop_words


def replace_century_words(text):
    """
    Replace specific century-related words in the input text with their predefined replacements.
    :param text: The input text as a string.
    :return: The processed text with century words replaced.
    """
    words = text.lower().split()
    for i in range(len(words)):
        if words[i] in century_words:
            words[i] = century_words[words[i]]
    return ' '.join(words)


def replace_additional_words(text):
    """
    Replace specific words in the input text with additional predefined replacements.
    :param text: The input text as a string.
    :return: The processed text with additional words replaced.
    """
    for word, replacement in additional_replacements.items():
        text = re.sub(r'\b' + word + r'\b', replacement, text)
    return text


def replace_number_words(text):
    """
    Replace numeric words in the input text with their corresponding numeric values.
    :param text: The input text as a string.
    :return: The processed text with number words replaced.
    """
    words = text.split()
    for i in range(len(words)):
        if words[i] in number_words:
            words[i] = number_words[words[i]]
    return ' '.join(words)


def remove_puncts(input_string):
    """
    Remove punctuation and replace certain dashes with spaces in the input string.
    :param input_string: The input string.
    :return: The processed string without punctuation.
    """
    return re.sub(r'[-–—]', ' ', input_string.translate(str.maketrans('', '', string.punctuation)).lower())


def fix_text_encoding(text):
    """
    Fix text encoding issues such as garbled characters.
    :param text: The input text as a string.
    :return: The processed text with encoding fixed.
    """
    return ftfy.fix_text(text)


def remove_special_characters(text):
    """
    Remove special characters like &, #, @, $, %, etc., from the input text.
    :param text: The input text as a string.
    :return: The processed text without special characters.
    """
    return re.sub(r'[\&\#\@\$\%\^\*\(\)\<\>\?\!\+\-]', '', text)


def remove_extra_spaces(text):
    """
    Remove extra spaces and strip leading/trailing spaces from the input text.
    :param text: The input text as a string.
    :return: The processed text with extra spaces removed.
    """
    return re.sub(r'\s+', ' ', text).strip()


def remove_urls(text):
    """
    Remove URLs from the input text.
    :param text: The input text as a string.
    :return: The processed text without URLs.
    """
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)


def remove_html_tags(text):
    """
    Remove HTML tags from the input text.
    :param text: The input text as a string.
    :return: The processed text without HTML tags.
    """
    return re.sub(r'<.*?>', '', text)


def remove_emojis(text):
    """
    Remove emojis from the input text.
    :param text: The input text as a string.
    :return: The processed text without emojis.
    """
    return emoji.replace_emoji(text, replace='')



def lemmatize_text(text):
    morph = pymorphy2.MorphAnalyzer(lang='uk')

    return ' '.join([morph.parse(word)[0].normal_form for word in text.split()])


def preprocess_text_embedded(text):
    """
    Perform a series of text preprocessing steps such as encoding fixes, removing tags, URLs, special characters,
    punctuation, emojis, replacing specific words, and r
    emoving extra spaces.
    :param text: The input text as a string.
    :return: The fully preprocessed text as a string.
    """
    text = fix_text_encoding(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_special_characters(text)
    text = remove_emojis(text)
    text = remove_puncts(text)
    text = replace_century_words(text)
    text = replace_additional_words(text)
    text = replace_number_words(text)
    text = remove_extra_spaces(text)
    tokens = [word for word in text.split() if word not in ukrainian_stop_words]
    return ' '.join(tokens)


def preprocess_text_frequency(text):
    """
    Perform a series of text preprocessing steps such as encoding fixes, removing tags, URLs, special characters,
    punctuation, emojis, replacing specific words, and r
    emoving extra spaces.
    :param text: The input text as a string.
    :return: The fully preprocessed text as a string.
    """
    text = fix_text_encoding(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_special_characters(text)
    text = remove_emojis(text)
    text = remove_puncts(text)
    text = replace_century_words(text)
    text = replace_additional_words(text)
    text = replace_number_words(text)
    text = remove_extra_spaces(text)
    tokens = [word for word in text.split() if word not in ukrainian_stop_words]
    text = lemmatize_text(text)
    return ' '.join(tokens)


def process_data_embedded(input_path, output_path):
    """
    Process textual data from a CSV file, apply preprocessing, and save the processed results to a new CSV file.
    :param input_path: The file path to the input CSV containing 'content' and 'news_id' columns.
    :param output_path: The file path to save the processed CSV.
    :return: None. Saves the processed file at the output path.
    """
    # Load the data
    documents_df = pd.read_csv(input_path)

    # Extract the text corpus
    corpus = documents_df['content'].tolist()

    # Preprocess texts in a single-threaded mode
    processed_corpus = [preprocess_text_embedded(text) for text in tqdm(corpus, total=len(corpus))]

    # Save the processed corpus to a CSV
    df = pd.DataFrame({'news_id': documents_df['news_id'], 'content': processed_corpus})
    df.to_csv(output_path, index=False)
    print(f"File {output_path} created successfully.")


def process_data_frequency(input_path, output_path):
    """
    Process textual data from a CSV file, apply preprocessing, and save the processed results to a new CSV file.
    :param input_path: The file path to the input CSV containing 'content' and 'news_id' columns.
    :param output_path: The file path to save the processed CSV.
    :return: None. Saves the processed file at the output path.
    """
    # Load the data
    documents_df = pd.read_csv(input_path)

    # Extract the text corpus
    corpus = documents_df['content'].tolist()

    # Preprocess texts in a single-threaded mode
    processed_corpus = [preprocess_text_frequency(text) for text in tqdm(corpus, total=len(corpus))]

    # Save the processed corpus to a CSV
    df = pd.DataFrame({'news_id': documents_df['news_id'], 'content': processed_corpus})
    df.to_csv(output_path, index=False)
    print(f"File {output_path} created successfully.")