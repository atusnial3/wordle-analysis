from data import ANSWERS, GUESSES, EVIL_ANSWERS, TOP_WORDS, WORST_WORDS
import numpy as np
import time
from tqdm import tqdm
from pprint import pprint
from multiset import Multiset
import itertools
from itertools import combinations
import random
from collections import Counter
from functools import reduce
import sys
import re
from joblib import Parallel, delayed
import contextlib
import joblib
import csv
import cProfile
import pstats

ALL_WORDS = GUESSES + ANSWERS

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager allowing tqdm usage in joblib parallelization.
    taken from stack overflow:
    https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# returns all possible patterns of length k with characters from the given set
# recursively
def possible_patterns(set=['B', 'Y', 'G'], k=5):
    """Returns a list of all possible length k patterns built from characters in set.

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

def dumb_entropy(ls, word_set=ANSWERS):
    """Filters search space to words that are potential answers.

    Returns a list of words that match the guess patterns given in ls, using an
    entropy measure that takes into account the positions of green, yellow, and
    black letters in a list of guesses.

    Args:
        ls: A list of dicts, where each dict contains the pattern and guess for
            a given guess. For example:
            [
                {'pattern' : ['B', 'B', 'B', 'Y', 'Y'], 'guess' : 'raise'},
                {'pattern' : ['G', 'B', 'B', 'B', 'G'], 'guess' : 'bumps'}
            ]
        word_set: A list of the search space of possible answers to narrow down
            using this entropy measure.

    Returns:
        A list of strings that are possible target words given the patterns and
        guesses in ls.
    """
    blacks = set()
    word = ['' for _ in range(5)]
    impossible = [set() for _ in range(5)]
    yellows = set()
    for dct in ls:
        pattern, guess = dct['pattern'], dct['guess']
        for i in range(5):
            if pattern[i] == 'B':
                set_imp = True
                for j in range(5):
                    if guess[j] == guess[i] and pattern[j] != 'B':
                        set_imp = False
                # blacks.add(guess[i])
                if set_imp:
                    [imp_set.add(guess[i]) for imp_set in impossible]
            elif pattern[i] == 'Y':
                impossible[i].add(guess[i])
                yellows.add(guess[i])
                # yellows.add(guess[i])
            else:
                word[i] = guess[i]
    # find words matching this information
    # regex match on word
    reg_arr = word
    for idx in range(5):
        if reg_arr[idx] == '':
            reg_arr[idx] = '[^'
            if len(impossible[idx]) > 0:
                reg_arr[idx] += ''.join([elem for elem in impossible[idx]])
            reg_arr[idx] += ']'
            if reg_arr[idx] == '[^]':
                reg_arr[idx] = '\w'
    reg = ''.join(reg_arr)

    p = re.compile(reg)

    # print(reg, yellows)
    return [word for word in word_set if p.match(word) and set(word) >= yellows]

# returns True if the guess matches the pattern of the info given
def match_info(guess, known, yellows, blacks, impossible):
    """Helper function used in smart_entropy returning True if guess matches the pattern."""
    for i in range(5):
        if (
            (known[i] != '' and guess[i] != known[i]) or
            (guess[i] in blacks) or
            (guess[i] in impossible[i])
           ):
            return False
    if len(yellows - Multiset(guess)) > 0:
        return False
    return True

def smart_entropy(ls, word_set=ANSWERS):
    """Filters search space to words that are potential answers.

    A less-accurate and less-efficient version of dumb_entropy used as backup
    logic.

    Args:
        ls: A list of dicts, where each dict contains the pattern and guess for
            a given guess. For example:
            [
                {'pattern' : ['B', 'B', 'B', 'Y', 'Y'], 'guess' : 'raise'},
                {'pattern' : ['G', 'B', 'B', 'B', 'G'], 'guess' : 'bumps'}
            ]
        word_set: A list of the search space of possible answers to narrow down
            using this entropy measure.

    Returns:
        A list of strings that are possible target words given the patterns and
        guesses in ls.
    """
    known = ['' for _ in range(5)] # any green seen will update the known word
    greens = Multiset()
    # multiset of letters, no idx (repeats are allowed & counted)
    yellows = Multiset()
    # set of letters, no idx
    blacks = set()
    # list of 5 lists, each containing all letters that can't go in that spot
    impossible = [[] for _ in range(5)]
    for dct in ls:
        yellows_in_guess = Multiset()
        for idx, val in enumerate(zip(dct['pattern'], dct['guess'])):
            pat, gus = val
            if pat == 'G':
                if known[idx] != gus:
                    greens.add(gus)
                known[idx] = gus
                yellows.discard(gus, multiplicity=1)
            elif pat == 'Y':
                if known[idx] == '':
                    yellows_in_guess.add(gus)
                    if gus not in impossible[idx] and impossible[idx] != '-':
                        impossible[idx].append(gus)
            else:
                if gus not in yellows and gus not in greens:
                    blacks.add(gus)
        to_remove = []
        for yel_gus in yellows_in_guess:
            if yel_gus in greens:
                to_remove += yel_gus
        for elem in to_remove:
            yellows_in_guess.discard(elem, greens[elem])
        yellows = yellows | yellows_in_guess

        # check impossible to see if you've finalized position of any yellows
        # by eliminating enough positions
        to_remove = []
        for elem, mult in yellows.items():
            ls = list(itertools.chain.from_iterable(impossible))
            # if 5 - the number of impossibles is mult, we
            # know where elem goes mult times and fill it in
            if mult + ls.count(elem) == 5:
                for i in range(5):
                    if known[i] == '' and elem not in impossible[i]:
                        known[i] = elem
                    elif elem in impossible[i]:
                        impossible[i].remove(elem)
                # remove from yellows since it's been filled in and add to greens
                to_remove.append(elem)
        for elem in to_remove:
            mult = yellows.remove(elem)
            greens.add(elem, mult)

        # replace stuff in impossible if the position has already been finalized
        for idx, val in enumerate(known):
            if val != '':
                impossible[idx] = ['-']
    return [guess for guess in word_set if match_info(guess, known, yellows, blacks, impossible)]


def average_elimination(guesses, existing=None, word_set=ANSWERS):
    """Computes the number of possible answers left after some guesses.

    Returns, for each possible target in word_set, the number of words left in
    the search space after filtering using the dumb_entropy measure. It uses
    smart_entropy as a backup measure if dumb_entropy ever gives 0 possiblities
    in the search space.

    Args:
        guesses: A string or list of strings of the sequence of guesses to try.
        existing: A list of dicts containing guesses and patterns that have
            already narrowed down the search space. For example:
            [
                {'pattern' : ['B', 'B', 'B', 'Y', 'Y'], 'guess' : 'raise'},
                {'pattern' : ['G', 'B', 'B', 'B', 'G'], 'guess' : 'bumps'}
            ]
            were already guessed, and now we want to perform some sequence of
            new guesses.
        word_set: A list of the search space of possible answers.

    Returns:
        A 1-d numpy array where the ith entry contains the number of words left
        in the search space if the target word was word_set[i].
    """
    ret = np.zeros(len(word_set))
    if existing is not None:
        # trim the search space by existing if that hasn't already been done
        word_set = dumb_entropy(existing, word_set)
    for idx, target in enumerate(word_set):
        ls = []
        if existing is None:
            if isinstance(guesses, list):
                ret[idx] = len(dumb_entropy([{'pattern' : score(guess, target), 'guess' : guess} for guess in guesses], word_set))
            elif isinstance(guesses, str):
                # print({'pattern' : score(guesses, target), 'guess' : guesses})
                ret[idx] = len(dumb_entropy([{'pattern' : score(guesses, target), 'guess' : guesses}], word_set))
            else:
                raise ValueError('guesses must be of type list or str')
        else:
            if not isinstance(existing, list):
                raise ValueError('existing must be of type list')
            else:
                if isinstance(guesses, list):
                    new = [{'pattern' : score(guess, target), 'guess' : guess} for guess in guesses]
                    entropy = len(dumb_entropy(existing + new, word_set))
                    if entropy == 0:
                        # print(guesses, existing, target)
                        entropy = len(smart_entropy(existing + new, word_set))
                    ret[idx] = entropy
                elif isinstance(guesses, str):
                    new = {'pattern' : score(guesses, target), 'guess' : guesses}
                    entropy = len(dumb_entropy(existing + [new], word_set))
                    if entropy == 0:
                        # print(guesses, existing, target)
                        entropy = len(smart_entropy(existing + [new], word_set))
                    ret[idx] = entropy
                else:
                    raise ValueError('guesses must be of type list or str')
    return ret

def elim_helper(guesses, existing=None, word_set=ANSWERS):
    """Computes statistics for the array output by average_elimination."""
    arr = average_elimination(guesses, existing, word_set)
    return guesses, np.mean(arr), np.max(arr), Counter(arr)[1]

def best_starting_word(guesses=ALL_WORDS, answers=ANSWERS, filename='best_guess.csv'):
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
    with tqdm_joblib(tqdm(desc='calc', total=len(guesses))) as progress_bar:
        x = Parallel(n_jobs=-1)(
                delayed(elim_helper)(guess, word_set=answers)
                for guess in guesses)
    data = sorted(x, key=lambda y: y[1])

    with open(filename, 'w', newline='') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['Guess', 'Mean', 'Max', 'Num_Solved'])
        for row in data:
            csv_out.writerow([row[0], row[1], row[2]])
    return data

def optimal_wordle(answer_set=ANSWERS):
    """CLI to play a game of Wordle optimally, for use while playing Wordle.

    Args:
        answer_set: A list of strings containing the possible Wordle answers.
    """
    ls = [] # list of dict [{pattern : pattern, guess : guess}]
    possible = [0, 0]
    print(f'The optimal first guess is \'{TOP_WORDS[0]}\'.')
    while len(possible) > 1:
        guess = input('Input your guess:\n')
        pattern = list(input('Input the color pattern using B, Y, and G (e.g. BBYGB)\n'))
        ls.append({'pattern' : pattern, 'guess' : guess})
        print('Computing possible target words... ', end='')
        possible = dumb_entropy(ls, answer_set)
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
        with tqdm_joblib(tqdm(desc='calc', total=len(ALL_WORDS))) as progress_bar:
            guess_ranks = Parallel(n_jobs=6)(
                delayed(elim_helper)(guess, ls, possible)
                for guess in ALL_WORDS)
        guess_ranks = sorted(guess_ranks, key=lambda y: y[1])
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

def best_second_word_by_pattern(first_guess='roate', filename='best_second_word_by_pattern.csv'):
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
    # best_words = [rank_second_guess(pattern, first_guess)
    #     for pattern in tqdm(patterns)]
    with tqdm_joblib(tqdm(total=len(patterns))) as progress_bar:
        best_words = Parallel(n_jobs=6)(
            delayed(rank_second_guess)(pattern, first_guess)
            for pattern in patterns)
    best_words = sorted(best_words, key=lambda y: y[5])
    pprint(best_words)

    return best_words

def rank_second_guess(pattern, first_guess):
    """Helper that ranks second guesses for a given pattern and first guess."""
    existing = [{'pattern' : list(pattern), 'guess' : first_guess}]
    possible = dumb_entropy(existing)
    if len(possible) > 1:
        start_time = time.time()
        guess_ranks = [elim_helper(guess, existing, possible)
            for guess in ALL_WORDS]
        # with tqdm_joblib(tqdm(total=len(ALL_WORDS), leave=False, desc=pattern)) as progress_bar:
        #     guess_ranks = Parallel(n_jobs=6)(
        #         delayed(elim_helper)(guess, existing, possible)
        #         for guess in ALL_WORDS)
        guess_ranks = sorted(guess_ranks, key=lambda y: y[1])
        elapsed = time.time() - start_time
        return (guess_ranks[0][0], guess_ranks[0][1], guess_ranks[0][2], guess_ranks[0][3], pattern, elapsed)
    elif len(possible) == 1:
        return (possible[0], 0, 0, 0, pattern, 0)
    else:
        return ('N/A', 0, 0, 0, pattern, 0)

def main():
    # data = evaluate_two_guess()
    # data = best_second_word_by_pattern()
    # optimal_wordle(answer_set=ANSWERS)
    start_time = time.time()
    pprint(elim_helper('roate'))
    print(f'elapsed: {time.time() - start_time}')
    # best_second_word_by_pattern()
    # words = best_starting_word()
    # words = best_starting_word(answers=EVIL_ANSWERS, filename='best_guess_evil.csv')
if __name__ == '__main__':
    main()
