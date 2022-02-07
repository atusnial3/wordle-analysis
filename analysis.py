"""Analysis for Wordle game.

This script allows the user to analyze certain ideas in Wordle. All of the
defined functions in this file can be run directly via the main method.

Functions:
    - best_starting_word - ranks the best starting Wordle words
    - best_second_word_by_pattern - ranks the best second word for each possible
        pattern returned after a given first word
    - wordle_solver - automatically solves a game of Wordle
    - main - the main function of the script

Typical usage example:
    ranked = best_starting_word(filename='best_first_guess.csv')
    best_words = best_second_word_by_pattern('raise', 'best_second_word.csv')
    num_guesses = wordle_solver(target='funky', verbose=True)
"""



from data import ANSWERS, ALL_WORDS
from patterns_grid import generate_pattern_grid, load_pattern_grid, get_pattern_grid
from util import possible_patterns
from wordle import rank_next_guess, filter_words, score
import time
import csv
from tqdm import tqdm

PATTERNS_DICT = dict()

def best_starting_word(guesses=ALL_WORDS, answers=ANSWERS, filename='outputs/best_guess_grid.csv'):
    """Ranks possible first guesses in Wordle by average number of words eliminated.

    Computes the number of words left in the search space after each guess for
    each possible answer, and computes three measures on these arrays: mean,
    max, and num_solved. Mean gives the average number of words left for the
    guess, max gives the worst-case number of words left for the guess, and
    num_solved gives the number of possible answers for which this guess will
    narrow the space down to 1 possible answer. Then ranks the words based on
    these three measures, and writes the output to a csv.

    Args:
        guesses: A list of strings containing the possible guesses to evaluate.
        answers: A list of strings containing the possible answers to evaluate.
        filename: A string representing the filename of the csv to write to.

    Returns:
        A list of tuples containing the possible guesses and their statistics
        according to the three measures. Each tuple looks something like:
        ('guess', 10.234, 15, 6) <- (guess, mean, max, num_solved)
    """

    guess_ranks = rank_next_guess(guesses, answers)

    with open(filename, 'w', newline='') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['Guess', 'Mean', 'Max', 'Num_Solved'])
        for row in guess_ranks:
            csv_out.writerow(row)
    return guess_ranks

def best_second_word_by_pattern(first_guess='roate', filename='outputs/best_second_word_by_pattern.csv'):
    """Computes the best second guess for each pattern given some first guess.

    Also writes the list of words and patterns to a csv file with filename
    Args:
        first_guess: A string representing the first Wordle guess.

    Returns:
        A list of tuples, where each tuple contains the guess, statistics for
        the guess, and the pattern corresponding to the guess. For example:
        ('phons', 2.368, 5.0, 8, 'YYBBG') <- (guess, mean, max, num_solved, pattern)
    """
    patterns = possible_patterns()
    best_words = []
    for pattern in tqdm(patterns):
        possible = filter_words(first_guess, pattern)
        if len(possible) > 1:
            best = rank_next_guess(ALL_WORDS, possible)[0][0]
            best_words.append((best, pattern))
        elif len(possible) == 1:
            best_words.append((possible[0], pattern))
        else:
            best_words.append(('N/A', pattern))

    return best_words


def wordle_solver(target='skill', verbose=False, answer_set=ANSWERS, first_guess='raise'):
    """Automatically solves a game of Wordle for a given first guess and target.

    Args:
        target: The len 5 string the solver is attempting to guess.
        verbose: A boolean determining whether to display the solver's steps.
        answer_set: The set of answers for the solver to consider.
        first_guess: The len 5 string that the solver uses as its first guess.

    Returns:
        The number of guesses it took to solve the game.
    """
    num_guesses = 0
    possible = answer_set
    best_guess = first_guess
    while len(possible) >= 1:
        guess = best_guess
        pattern = score(best_guess, target)
        num_guesses += 1
        if verbose:
            print(f'Guess {num_guesses}: {guess}')
            print(f'Pattern: {"".join(pattern)}')
        if pattern == list('GGGGG'):
            if verbose:
                print(f'Answer: {possible[0]} ({num_guesses} guesses)')
            return num_guesses
        possible = filter_words(guess, pattern, possible)
        if verbose:
            print(f'{len(possible)} possible targets')
        if len(possible) == 1:
            best_guess = possible[0]
        else:
            guess_ranks = rank_next_guess(ALL_WORDS, possible)
            best_guess = guess_ranks[0][0]
    raise RuntimeError(f'Wrong guess: target {target}, actual {possible}')

def main():
    start_time = time.time()
    # best_starting_word()
    # best_second_word_by_pattern()
    wordle_solver(verbose=True)
    elapsed = time.time() - start_time
    print(f'{elapsed} seconds elapsed')

if __name__ == '__main__':
    load_pattern_grid()
    main()
