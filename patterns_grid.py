"""Generates and loads the pattern grid.

Functions in this module work to maintain a grid of patterns for easy lookup,
avoiding high time cost in parsing many guesses at once. The grid generated
is saved to a file as a numpy array, and loaded into memory as a dictionary
at runtime in scripts in this project. The grid is a 2D numpy array with shape
(N, N) where N = len(ALL_WORDS) is the number of total words in the dataset.
The [i, j] entry of the grid contains the unique integer ID for the pattern of
B, Y, and G that Wordle would give if ALL_WORDS[i] was guessed and ALL_WORDS[j]
was the target answer. The design is inspired by 3B1B's grid solution.

Functions:
    - generate_pattern_grid - generates the full N by N grid of patterns
    - load_pattern_grid - loads the pattern grid to a dictionary at runtime
    - get_pattern_grid - gets a subset of the pattern grid.

Typical usage example:
    grid = generate_pattern_grid()
    PATTERNS_DICT = load_pattern_grid()
    grid_subset = get_pattern_grid(ALL_GUESSES, possible_targets)
"""

from data import ALL_WORDS
from scoring import score
from util import pattern_to_num
from os.path import exists
import itertools as it
import numpy as np
import numpy.typing as npt
from tqdm import tqdm


def generate_pattern_grid(filename: str = 'patterns_grid.npy') -> npt.NDArray:
    """Generates the grid of patterns from the entire corpus.

    This function only needs to be called once as it saves the pattern grid to
    a .npy file as a 2D numpy array.

    Args:
        filename: A string containing the filename to save the pattern grid to.

    Returns:
        A 2D numpy array containing the pattern grid. The [i,j] position in the
        array holds the pattern that Wordle would give if ANSWERS[i] is guessed
        and ANSWERS[j] is the target word.
    """
    patterns = np.zeros((len(ALL_WORDS), len(ALL_WORDS)))
    for i, w1 in enumerate(tqdm(ALL_WORDS)):
        for j, w2 in enumerate(ALL_WORDS):
            patterns[i, j] = pattern_to_num(score(w1, w2))

    np.save(filename, patterns)
    return patterns


def load_pattern_grid(filename: str = 'patterns_grid.npy') -> dict:
    """Loads the pattern grid at runtime.

    If the file path given is not found, it (re)generates the grid calling the
    generate_pattern_grid function. It loads the global variable PATTERNS_DICT
    with the pattern grid and with another dict that indexes the entire set of
    words.

    Args:
        filename: A string containing the path to load the pattern grid from.

    Returns:
        A dictionary patterns_dict, where patterns_dict['grid'] contains the
        patterns grid, and patterns_dict['word_index'] is a dict indexing the
        words in ALL_WORDS.
    """
    patterns_dict = dict()
    # if the grid doesn't exist, generate it
    if not exists(filename):
        _ = generate_pattern_grid(filename)
    patterns_dict['grid'] = np.load(filename)
    patterns_dict['word_index'] = dict(zip(ALL_WORDS, it.count()))
    return patterns_dict


def get_pattern_grid(words1: list[str],
                     words2: list[str],
                     patterns_dict: dict = None) -> npt.NDArray:
    """Gets the pattern grid for the cross product of words1 and words2.

    That is, it gets the subset of the grid PATTERNS_DCT['grid'] where the
    indices on each axis are the indices of words1 and words2 stored in
    PATTERNS_DICT['word_index'].

    Args:
        words1: A list of strings to get pattern matches of.
        words2: A list of strings to get pattern matches of.
        patterns_dict: The full patterns dictionary

    Returns:
        The subset of pattern dict containing every pattern that would result
        from a guess of w1 and a target of w2 for all w1 in words1 and w2 in
        words2.
    """
    if patterns_dict is None:
        patterns_dict = load_pattern_grid()
    indices1 = [patterns_dict['word_index'][w] for w in words1]
    indices2 = [patterns_dict['word_index'][w] for w in words2]
    return patterns_dict['grid'][np.ix_(indices1, indices2)]
