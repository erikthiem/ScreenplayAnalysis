# rnn_trainer.py
# Code that trains on the Screenplay data and produces a model for text generation.

import sys

from Preprocess import ScreenplayData

class RNN_Trainer:
	def __init__(self, inFile):
		
if __name__ == "__main__":
	vocab_size = sys.argv[1] if len(sys.argv) > 1 else 8000
	train = ScreenplayData("../data/processed/lines_by_genre.db", vocab_size)