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
from pprint import pprint
import csv


def generate_word_frequencies(filename: str = 'outputs/word_frequencies.csv',
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


def load_word_frequencies(filename: str = 'outputs/word_frequencies.csv') \
        -> list[dict[str, tuple[float, float, float]]]:
    """Loads word frequencies into a list of dictionaries from file at runtime.

    Returns the list of dicts of words and their frequency measures, which
    is a list of dictionaries that looks like:
    [{'Word' : 'about', 'Frequency' : 0.00251, 'ZipF': 6.40, Sigmoid: .99932},
     {'word' : 'their', 'Frequency' : 0.00214, 'ZipF': 6.33, Sigmoid: .99913},
     {'word' : 'there', 'Frequency' : 0.00204, 'ZipF': 6.31, Sigmoid: .99907}]

    Args:
        filename: the path to the file to load from

    Returns:
        The list of dicts of words and frequency measures
    """
    if not exists(filename):
        generate_word_frequencies(filename=filename)

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        freq_list = list(reader)
        return freq_list


def main():
    freq_list = load_word_frequencies()
    pprint(freq_list[:20])


if __name__ == "__main__":
    main()
