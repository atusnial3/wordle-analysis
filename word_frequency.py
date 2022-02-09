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
import csv

global WORD_FREQ_LIST


def generate_word_frequencies(filename='outputs/word_frequencies.csv',
                              k=3.5, midpoint=3, L=1):
    """Generates a csv of the frequencies of all words in the corpus.

    Gives both the actual frequency given by word_frequency, as well as the
    logarithmic frequency from zipf_frequency. Also computes the logistic
    function of the zipf_frequencies using the given parameters.

    Args:
        filename: The path to the file to write to
        k: The steepness parameter for logistic function
        midpoint: The midpoint parameter for the logistic function
        L: The scale parameter for the logistic function
    """
    with open(filename, 'w', newline='') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['Word', 'Freq', 'ZipF', 'Sigmoid'])
        for word in tqdm(ALL_WORDS):
            freq = word_frequency(word)
            zipf = zipf_frequency(word)
            sigmoid = logistic(x=zipf, k=k, midpoint=midpoint, L=L)
            csv_out.writerow([word, freq, zipf, sigmoid])


def load_word_frequencies(filename='outputs/word_frequencies.csv'):
    """Loads word frequencies into a list of dictionaries from file at runtime.

    Word frequencies are loaded into the global variable WORD_FREQ_LIST, which
    is a list of dictionaries that looks like:
    [
    ,,6.4,0.999993209641302
    their,0.00214,6.33,0.9999913245093576
    there,0.00204,6.31,0.9999906954711627

        {'word' : 'about', 'freq' : 0.00251, }
    ]

    Args:
        filename: the path to the file to load from
    """
    global WORD_FREQ_LIST
    if not exists(filename):
        generate_word_frequencies(filename=filename)

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        WORD_FREQ_LIST = list(reader)


def main():
    load_word_frequencies()
    print(WORD_FREQ_LIST)


if __name__ == "__main__":
    main()
