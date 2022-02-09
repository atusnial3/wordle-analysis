"""Analyzes the entropy of a Wordle system based on frequency analysis of data.

Contains functions to generate and load word frequency data.

Functions:
    - generate_word_frequencies - generates the csv of word frequencies
    - load_word_frequencies - loads word frequencies list into memory at
        runtime
"""

from util import logistic
from data import ALL_WORDS
from os.path import exists
from wordfreq import word_frequency, zipf_frequency
from tqdm import tqdm
import pandas as pd
import csv


def generate_word_frequencies(filename: str = 'resources/word_frequencies.csv',
                              k: float = 3.5,
                              midpoint: float = 3,
                              scale: float = 1):
    """Generates a csv of the frequencies of all words in the corpus.

    Gives both the actual frequency given by word_frequency, and the
    logarithmic frequency from zipf_frequency. Also computes the logistic
    function of the zipf_frequencies using the given parameters.

    Args:
        filename: The path to the file to write to
        k: The steepness parameter for logistic function
        midpoint: The midpoint parameter for the logistic function
        scale: The scale parameter for the logistic function
    """
    with open(filename, 'w', newline='') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['Word', 'Freq', 'ZipF', 'Sigmoid'])
        for word in tqdm(ALL_WORDS):
            freq = word_frequency(word, 'en')
            zipf = zipf_frequency(word, 'en')
            sigmoid = logistic(zipf, k, midpoint, scale)
            csv_out.writerow([word, freq, zipf, sigmoid])


def load_word_frequencies(filename: str = 'resources/word_frequencies.csv',
                          load_format: str = 'df') \
        -> list[dict[str, tuple[float, float, float]]]:
    """Loads word frequencies into a list of dictionaries from file at runtime.

    Returns the list of dicts of words and their frequency measures, which
    is a list of dictionaries that looks like:
    [{'Word' : 'about', 'Frequency' : 0.00251, 'ZipF': 6.40, Sigmoid: .99932},
     {'word' : 'their', 'Frequency' : 0.00214, 'ZipF': 6.33, Sigmoid: .99913},
     {'word' : 'there', 'Frequency' : 0.00204, 'ZipF': 6.31, Sigmoid: .99907}]
    if load_format is dict, and a list of lists that looks like:
    [['about', .00251, 6.40, .99932],
     ['their', .00214, 6.33, .99913],
     ['there', .00204, 6.31, .99907]] if load_format is list, and a pandas
    DataFrame if load_format is 'df'

    Args:
        filename: the path to the file to load from
        load_format: 'df', 'list' or 'dict' for the format of each row
            in the output

    Returns:
        The list of dicts of words and frequency measures
    """
    if load_format not in ['list', 'dict', 'df']:
        raise ValueError('load_format must be either list or dict')

    if not exists(filename):
        generate_word_frequencies(filename=filename)

    if load_format == 'df':
        return pd.read_csv(filename)

    with open(filename, 'r') as f:
        reader = csv.DictReader(f) if load_format == 'dict' else csv.reader(f)
        freq_list = list(reader)
        return freq_list
