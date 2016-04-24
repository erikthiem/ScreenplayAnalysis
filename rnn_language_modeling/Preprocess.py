# preprocess.py
# Code to read and preprocess data.

import sqlite3
import nltk
import itertools
import numpy as np

class ScreenplayData:
	""" Reads in data from SQL table... """
	def __init__(self, database_path, vocabulary_size=8000):
		# Connect to the SQL table
		conn = sqlite3.connect(database_path)
		c = conn.cursor()

		# Selects all lines where the genre contains 'action' or 'adventure.'
		c.execute('SELECT line FROM lines WHERE genres like "%action%" or genres like "%adventure%";')
		
		# Read all lines across all genres.
		# c.execute('SELECT line FROM lines;')

		# Results are in a tuple of size 1.
		entries = c.fetchall()
		conn.commit()

		# Close the connection.
		conn.close()

		# Retrieving each of the sentences.
		sentences = [str(entry[0]) for entry in entries]

		# Tokenize the sentences into words
		tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

		# Count the word frequencies
		word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
		print("Found %d unique word tokens." % len(word_freq.items()))

		unknown_token = "UNKNOWN_TOKEN"
		sentence_start_token = "SENTENCE_START"
		sentence_end_token = "SENTENCE_END"

		# Get the most common words and build index_to_word and word_to_index vectors.
		vocab = word_freq.most_common(vocabulary_size-1)
		index_to_word = [x[0] for x in vocab]
		index_to_word.append(unknown_token)
		word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

		print("Using vocabulary size %d." % vocabulary_size)
		# print("The least frequent word in our vocabulary is '%s'. It appeared %d times." \
		# 	% (vocab[-1][0], vocab[-1][1]))

		# Replace all words not in our vocabulary with the unknown token.
		for i, sent in enumerate(tokenized_sentences):
		    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

		# Create the training data
		self.X = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
		self.Y = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

if __name__ == "__main__":
	data = ScreenplayData("../data/processed/lines_by_genre.db")
	# print data.sentences
