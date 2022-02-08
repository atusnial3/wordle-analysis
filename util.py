"""Utility functions to facilitate functions from other files in the project.

Functions:
    - possible_patterns - outputs a list of all possible patterns
    - chars_to_int_gen - converts characters B, Y, G to 0, 1, 2 respectively
    - pattern_to_num - converts a pattern to its unique integer ID
    - num_to_pattern - converts a unique integer ID to its pattern
    - count_unique_by_row - gives rowwise the counts of unique entries in a 2D array
    - logistic - computes the logistic function at x with given parameters
"""

import numpy as np
import math

def possible_patterns(set=['B', 'Y', 'G'], k=5):
    """Return a list of all possible length k patterns built from characters in set.

    Args:
        set: The set of characters to build patterns from
            (default {'B', 'Y', 'G'})
        k: The length of the patterns to build (default 5)

    Returns:
        A list of size math.pow(len(set), k) of patterns, where each pattern
        is a string of length k.
    """
    ls = []
    possible_patters_rec(set, "", k, ls)
    return ls

def possible_patters_rec(set, word, k, ls):
    """Underlying recursive function called by possible_patterns"""
    if (k == 0) :
        ls.append(word)
        return
    for char in set:
        possible_patters_rec(set, word + char, k - 1, ls)

def chars_to_int_gen(pattern):
    """Generator that converts 'B', 'Y', and 'G' to 0, 1, and 2 respectively.

    Used as a helper in pattern_to_num.

    Args:
        pattern: A list of characters containing the pattern to convert.
    Returns:
        A generator containing the sequence of integers corresponding to the
        pattern's characters.
    """
    for char in pattern:
        if char == 'B':
            yield 0
        elif char == 'Y':
            yield 1
        else:
            yield 2

def pattern_to_num(pattern):
    """Take in a pattern and return a unique integer corresponding to it.

    Uses a ternary expansion to associate a unique ID with each pattern. The ith
    character is 3^i * x, where x is 0, 1, or 2 corresponding to B, Y, and G
    respectively.

    Args:
        pattern: A list of characters containing the pattern to convert.

    Returns:
        The ternary integer representation of the pattern.
    """
    return sum(
        (3 ** i) * val for i, val in enumerate(chars_to_int_gen(pattern))
    )

def num_to_pattern(num):
    """Take in an integer from 0 to 242 and return its unique pattern.

    Uses a ternary expansion to associate a pattern with each unique integer ID,
    where we know the ith character in the pattern is 3^i * x, where x is 0, 1,
    or 2 corresponding to B, Y, and G respectively.

    Args:
        num: An integer from 0 to 242 representing the ID to convert.

    Returns:
        The pattern corresponding to this unique ID.
    """
    dct = {0 : 'B', 1 : 'Y', 2 : 'G'}
    pattern = ''
    for i in range(5):
        num, remainder = divmod(num, 3)
        pattern += dct[remainder]
    return pattern

def count_unique_by_row(a):
    """Give the number of unique entries rowwise in an array (helper function).

    Adapted from https://stackoverflow.com/a/28789027. The output puts the
    count of each unique element (rowwise) in the array into the location of its
    first occurrence in the row. The function uses a complex number trick to
    avoid repeated calls to np.unique - it adds a unique imaginary number to
    each row (so elements from different rows correspond to different
    complex numbers. Then, a single call to np.unique isolates unique elements
    by row correctly. This is used as a helper function in rank_next_guess

    Args:
        a: A 2D numpy array to get counts of unique values from

    Returns:
        A 2D numpy array where the first occurrence of a unique value in a is
        replaced by the count of that value in that row of a.
    """
    weight = 1j*np.linspace(0, a.shape[1], a.shape[0], endpoint=False)
    b = a + weight[:, np.newaxis]
    u, ind, cnt = np.unique(b, return_index=True, return_counts=True)
    b = np.zeros_like(a)
    np.put(b, ind, cnt)
    return b

def logistic(x, k=3.5, midpoint=3, L=1):
    """Evaluates the logistic function with given parameters at x.

    The logistic function is given by
        L / (1 + e^(-k(x - midpoint)))

    Args:
        k: A float determining the steepness of the sigmoid curve - higher
            values -> steeper curve
        midpoint: A float determining the midpoint of the curve (the x values
            for which logistic(x) = 0.5)
        L: A float determining the scale of the sigmoid curve - L is the upper
            bound of the function as x -> \infty
    """
    return L / (1 + math.exp((-1 * k) * (x - midpoint)))
