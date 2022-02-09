# -*- coding: utf-8 -*-
"""Functions to score Wordle guesses.

Functions:
    - score_simple - fast logic to score words without double letters
    - score - computes the pattern Wordle gives for any guess and target
"""


def score_simple(guess: str, target: str) -> list:
    """Simple scoring function, incorrect for words with double letters

    Used as a helper in score to optimize scoring speed for guesses without
    double letters.

    Args:
        guess: A length 5 string representing the guess to score
        target: A length 5 string representing the target answer to score by

    Returns:
        A length 5 list of characters from {'B', 'G', 'Y'} representing black,
        green, and yellow respectively, as the same pattern of colors Wordle
        would output for the given guess and target.
    """
    pattern = []
    for g, t in zip(guess, target):
        if g == t:
            pattern.append('G')
        elif g in target:
            pattern.append('Y')
        else:
            pattern.append('B')
    return pattern


def score(guess: str, target: str) -> list:
    """Scores a guess the same way Wordle would score it.

    Returns the pattern of greens, yellows, and blacks that Wordle would output
    for a given guess and target answer.

    Args:
        guess: A length 5 string representing the guess to score
        target: A length 5 string representing the target answer to score by

    Returns:
        A length 5 list of characters from {'B', 'G', 'Y'} representing black,
        green, and yellow respectively, as the same pattern of colors Wordle
        would output for the given guess and target.
    """
    # default to simple logic
    if len(set(guess)) == len(guess) and len(set(target)) == len(target):
        return score_simple(guess, target)
    pattern = [''] * 5
    already_matched = []
    # green loop - evaluate greens first
    for idx, val in enumerate(zip(guess, target)):
        g, t = val
        if g == t:
            pattern[idx] = 'G'
            already_matched.append(g)

    # yellow/black loop
    for idx, val in enumerate(zip(guess, target)):
        # only look at letters that haven't yet been given a match
        if pattern[idx] == '':
            g, t = val
            # will be yellow if g is in target and all instances of g
            # have not already been matched
            if (g in target and
                    (g not in already_matched or
                     already_matched.count(g) < target.count(g))):
                pattern[idx] = 'Y'
                already_matched.append(g)
            else:
                pattern[idx] = 'B'
    return pattern
