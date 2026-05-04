import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


CUSTOM_STOPWORDS = {"www", "https", "s", "t", "u"}
BASE_STOPWORDS = set(ENGLISH_STOP_WORDS) - {'not', 'no', 'nor', 'none', 'neither', 'never'}
STOPWORDS = BASE_STOPWORDS.union(CUSTOM_STOPWORDS)


def remove_stopwords(sentence):
    words = sentence.split()
    clean_words = [w for w in words if w not in STOPWORDS]

    return " ".join(clean_words)


def replace_numbers(match):
    token = match.group(0).strip()
    if token.startswith('-'):
        return ' neg_num_token '
    return ' pos_num_token '

def replace_percentages(match):
    token = match.group(0).strip()
    if token.startswith('-'):
        return ' neg_perc_token '
    return ' pos_perc_token '

def clean_numbers(text):
    text = re.sub(r'-?\s?\d+(\.\d+)?\s?%', replace_percentages, text)
    text = re.sub(r'-?\s?\d+(\.\d+)?', replace_numbers, text)

    return text

def clean_interpunction(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def clean_text(text):
    text = text.lower()
    text = clean_numbers(text)
    text = clean_interpunction(text)
    text = remove_stopwords(text)
    return text
