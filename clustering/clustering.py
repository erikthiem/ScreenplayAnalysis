################################################
# An implementation of the Brown clustering    #
# algorithm.                                   #
# This file outputs each word with its         #
# bitstring, sorted by bitstring.              #
################################################

import sqlite3
import os
from nltk.tokenize import word_tokenize
from collections import defaultdict
import pickle

class Cluster():
    
    def __init__(self, first_word):
        self.words = []
        self.words.append(first_word)

    def addWord(self, word):
        self.words.append(word)

    def listWords(self):
        return self.words


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# File information
database_file_name = 'lines_by_genre.db'
database_folder_relative_path = '../data/processed/'

# Error checking for database file existence
database = database_folder_relative_path + database_file_name
if not os.path.isfile(database):
    print("\nError. Database {0} does not exist. Exiting!\n".format(database))
    exit(-1)

# Connect to the SQL table
conn = sqlite3.connect(database)
c = conn.cursor()
c.execute('SELECT * FROM lines;')
entries = c.fetchall()
conn.commit()
conn.close()

lines = [entry[1] for entry in entries]
'''
vocab = {}

count = 0

for index, item in enumerate(lines[0:100]):
    words = word_tokenize(item.lower())
    for word in words:
        vocab[word] = index
    count += 1

vocab_size = len(vocab)
print vocab_size
'''
# See if word frequencies were previously saved
if os.path.isfile("sorted_word_frequencies.pkl"):
    sorted_word_frequencies = load_obj("sorted_word_frequencies")
else:
    word_frequencies = defaultdict(int)
    for line in lines:
        words = word_tokenize(line.lower())
        for word in words:
            word_frequencies[word] += 1

    sorted_word_frequencies = sorted(word_frequencies.items(), reverse=True, key=lambda item: item[1])

    # Save word frequencies for future use
    save_obj(sorted_word_frequencies, "sorted_word_frequencies")
 
for pair in sorted_word_frequencies[0:10]:
    print("{0} {1}".format(pair[0], pair[1]))

# Create starting clusters for each of the first 1000 most-common words
clusters = []
num_clusters = 1000
for i in range(num_clusters):
    c = Cluster(sorted_word_frequencies[i][0])
    clusters.append(c)
