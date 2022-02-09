"""Contains functions to play Wordle and perform Wordle-specific tasks.

Functions in this script enable all Wordle logic, including scoring, filtering/
pattern-matching, and guess ranking. The function called in the main method of
the script provides a CLI to be used while playing a game of Wordle.

Functions:
    - filter_words - filters the word_set to match the given guess and pattern
    - rank_next_guess - ranks possible guesses by minimizing the average number
        of remaining words
    - play_wordle - CLI to play a game of Wordle
    - main - the main function of the script

Typical usage example:
    play_wordle(ANSWERS)
"""
from data import ANSWERS, ALL_WORDS
from patterns_grid import load_pattern_grid, get_pattern_grid
from util import count_unique_by_row, pattern_to_num
from pprint import pprint
import numpy as np
import time
import operator as op


def filter_words(guess: str,
                 pattern: list,
                 word_set: list[str] = ANSWERS,
                 patterns_dict: dict = None) -> list[str]:
    """Filter word_set to those words that match the given guess and pattern.

    Uses the pattern grid to find targets for which the given pattern would
    occur for this guess, and returns a list of those words.

    Args:
        guess: A len 5 string representing the word guessed.
        pattern: A len 5 list of characters from {'B', 'Y', 'G'} representing
            the pattern given for this guess and some unknown target.
        word_set: A list of strings representing the set all possible targets
            belong to.
        patterns_dict: The full patterns dictionary

    Returns:
        A list of strings that match the given guess and pattern. The list is a
        subset of word_set.
    """
    if patterns_dict is None:
        patterns_dict = load_pattern_grid()
    if isinstance(pattern, str):
        pattern = list(pattern)
    num = pattern_to_num(pattern)
    patterns = get_pattern_grid([guess], word_set, patterns_dict).flatten()
    return list(np.array(word_set)[patterns == num])


def rank_next_guess(word_set: list[str] = ALL_WORDS,
                    possible: list[str] = ANSWERS,
                    patterns_dict: dict = None)\
        -> list[tuple[str, float, float, float]]:
    """Ranks the possible next guesses from the word_set.

    Rank guesses by one of three metrics:
        (1) minimizing average number of possible targets remaining
        (2) minimizing maximum number of possible targets remaining
        (3) maximizing the number of targets the guess will fully solve
    By default, the guesses are sorted by minimum average, but can be sorted
    outside the function by one of the other metrics.

    Args:
        word_set: A list of strings containing all possible guesses.
        possible: A list of strings containing all currently possible targets.
        patterns_dict: The full patterns dictionary

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
    if patterns_dict is None:
        patterns_dict = load_pattern_grid()
    grid = get_pattern_grid(word_set, possible, patterns_dict)
    unq = count_unique_by_row(grid)
    averages = np.sum(unq, axis=1) / np.count_nonzero(unq, axis=1)
    worst_case = np.max(unq, axis=1)
    num_solved = np.count_nonzero(unq == 1, axis=1)
    guess_ranks = list(zip(word_set, list(averages), list(worst_case),
                           list(num_solved)))
    # sort 3 times - tertiary, secondary, primary. Have to do like this because
    # we need to sort by num_solved in reverse order
    # tertiary sort - worst_case
    guess_ranks.sort(key=op.itemgetter(2))
    # secondary sort - num_solved
    guess_ranks.sort(key=op.itemgetter(3), reverse=True)
    # primary sort - average left
    guess_ranks.sort(key=op.itemgetter(1))
    return guess_ranks


def play_wordle(answer_set: list[str] = ANSWERS,
                patterns_dict: dict = None):
    """CLI to play a game of Wordle optimally, for use while playing Wordle.

    Gives feedback after each guess ranking possible next guesses, but lets the
    user decide which guess to make at each choice.

    Args:
        answer_set: A list of strings containing the possible Wordle answers.
        patterns_dict: The full patterns dictionary
    """
    if patterns_dict is None:
        print('loading from scratch')
        patterns_dict = load_pattern_grid()
    possible = answer_set
    while len(possible) > 1:
        guess = input('Input your guess:\n')
        pattern = list(input('Input the color pattern using B, Y, and G '
                             '(e.g. BBYGB)\n'))
        print('Computing possible target words... ', end='')
        possible = filter_words(guess, pattern, possible, patterns_dict)
        print('Done.')
        if len(possible) <= 1:
            break
        print(f'Possible target words: ({len(possible)} words)')
        if len(possible) > 10:
            pprint(possible[:10] + ['...'])
        else:
            pprint(possible)

        print('Evaluating possible guesses:')
        start_time = time.time()
        guess_ranks = rank_next_guess(ALL_WORDS, possible, patterns_dict)
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
    main()
