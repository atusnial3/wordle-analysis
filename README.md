# wordle-analysis
Find optimal strategy for wordle &amp; wordle variants.

To use: clone the repository and install the necessary packages:
  - numpy
  - pandas
  - tqdm

The analysis.py and wordle.py files are scripts that can be run directly. Running wordle.py will open a CLI to play a game of Wordle. There are three functions in analysis.py that can be run, each analyzing some Wordle strategy.

Upon first running either script, the program will generate and save a large 2D numpy array referred to as the patterns grid. The [i, j] entry of the patterns grid stores the pattern Wordle would output if the ith word in the dataset was guessed and the jth word in the dataset was the target word. The patterns grid is stored as a .npy file, so it only needs to be generated once. It is loaded into runtime at the beginning of each script, dramatically speeding up the solver's computation speeds (from hours to seconds).
