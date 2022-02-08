"""Contains functions to play Wordle and perform Wordle-specific tasks.

Functions in this script enable all Wordle logic, including scoring, filtering/
pattern-matching, and guess ranking. The function called in the main method of
the script provides a CLI to be used while playing a game of Wordle.

Functions:
    - score - computes the pattern Wordle gives for the given guess and target
    - filter_words - filters the word_set to match the given guess and pattern
    - rank_next_guess - ranks possible guesses by minimizing the average number
        of remaining words
    - play_wordle - CLI to play a game of Wordle
    - main - the main function of the script

Typical usage example:
    play_wordle(ANSWERS)
"""
from data import ANSWERS, ALL_WORDS
from patterns_grid import generate_pattern_grid, load_pattern_grid, get_pattern_grid
from util import count_unique_by_row, pattern_to_num
from pprint import pprint
import numpy as np
import time
import operator as op

PATTERN_GRID = dict()

def score_simple(guess, target):
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


def score(guess, target):
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
            if g in target and (g not in already_matched or already_matched.count(g) < target.count(g)):
                pattern[idx] = 'Y'
                already_matched.append(g)
            else:
                pattern[idx] = 'B'
    return pattern

def filter_words(guess, pattern, word_set=ANSWERS):
    """Filter word_set to those words that match the given guess and pattern.

    Uses the pattern grid to find targets for which the given pattern would
    occur for this guess, and returns a list of those words.

    Args:
        guess: A len 5 string representing the word guessed.
        pattern: A len 5 list of characters from {'B', 'Y', 'G'} representing
            the pattern given for this guess and some unknown target.
        word_set: A list of strings representing the set all possible targets
            belong to.

    Returns:
        A list of strings that match the given guess and pattern. The list is a
        subset of word_set.
    """
    global PATTERN_DICT
    if isinstance(pattern, str):
        pattern = list(pattern)
    num = pattern_to_num(pattern)
    patterns = get_pattern_grid([guess], word_set).flatten()
    return list(np.array(word_set)[patterns == num])

def rank_next_guess(word_set=ALL_WORDS, possible=ANSWERS):
    """Ranks the possible next guesses from the word_set.

    Ranks guesses by one of three metrics:
        (1) minimizing average number of possible targets remaining
        (2) minimizing maximum number of possible targets remaining
        (3) maximizing the number of targets the guess will fully solve
    By default, the guesses are sorted by minimum average, but can be sorted
    outside the function by one of the other metrics.

    Args:
        word_set: A list of strings containing all possible guesses.
        possible: A list of strings containing all currently possible targets.

    Returns:
        A list of tuples, each containing a guess and its associated metrics.
        For example: [
            ('dines', 3.3235, 33.0, 19),
            ('pines', 3.3235, 33.0, 19),
            ('sinew', 3.3235, 34.0, 20),
            ('piles', 3.5313, 29.0, 16),
            ...
        ]
    """
    grid = get_pattern_grid(ALL_WORDS, possible)
    unq = count_unique_by_row(grid)
    averages = np.sum(unq, axis=1) / np.count_nonzero(unq, axis=1)
    worst_case = np.max(unq, axis=1)
    num_solved = np.count_nonzero(unq == 1, axis=1)
    guess_ranks = list(zip(ALL_WORDS, list(averages), list(worst_case), list(num_solved)))
    # sort 3 times - tertiary, secondary, primary. Have to do like this because
    # we need to sort by num_solved in reverse order
    # tertiary sort - worst_case
    guess_ranks.sort(key=op.itemgetter(2))
    # secondary sort - num_solved
    guess_ranks.sort(key=op.itemgetter(3), reverse=True)
    # primary sort - average left
    guess_ranks.sort(key=op.itemgetter(1))
    return guess_ranks

def play_wordle(answer_set=ANSWERS):
    """CLI to play a game of Wordle optimally, for use while playing Wordle.

    Gives feedback after each guess ranking possible next guesses, but lets the
    user decide which guess to make at each choice.

    Args:
        answer_set: A list of strings containing the possible Wordle answers.
    """
    possible = answer_set
    while len(possible) > 1:
        guess = input('Input your guess:\n')
        pattern = list(input('Input the color pattern using B, Y, and G (e.g. BBYGB)\n'))
        print('Computing possible target words... ', end='')
        possible = filter_words(guess, pattern, possible)
        print('Done.')
        if len(possible) <= 1:
            break
        print(f'Possible target words: ({len(possible)} words)')
        if len(possible) > 10:
            pprint(possible[:10] + ['...'])
        else:
            pprint(possible)

        print('Evaluating possible guesses:')
        guess_ranks = []
        start_time = time.time()
        guess_ranks = rank_next_guess(ALL_WORDS, possible)
        elapsed = time.time() - start_time
        print(f'Done. {elapsed} seconds')
        if len(guess_ranks) >= 10:
            print('The top guesses by average entropy are:')
            pprint(guess_ranks[:10])
            print('The top guesses by fewest max remaining are:')
            pprint(sorted(guess_ranks, key=lambda y: y[2])[:10])
            print('The top guesses by most totally solved are:')
            pprint(sorted(guess_ranks, key=lambda y: y[3], reverse=True)[:10])
        else:
            print('The top guesses by average entropy are:')
            pprint(guess_ranks)
            print('The top guesses by fewest max remaining are:')
            pprint(sorted(guess_ranks, key=lambda y: y[2]))
            print('The top guesses by most totally solved are:')
            pprint(sorted(guess_ranks, key=lambda y: y[3], reverse=True))

    if len(possible) == 0:
        print("You messed something up, there is no such match.")
    else:
        print(f'The word is {possible[0]}')

def main():
    play_wordle()

if __name__ == '__main__':
    load_pattern_grid()
    main()
