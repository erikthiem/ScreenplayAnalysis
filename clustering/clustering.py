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
import random

class Cluster():
    
    def __init__(self, first_word):
        self.words = []
        self.words.append(first_word)

    def addWord(self, word):
        self.words.append(word)

    def merge(self, other_cluster):
        for word in other_cluster.words:
            self.words.append(word)

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Returns the "quality" measure between two clusters
def qualityOfMerge(c1, c2):
    return random.random()


# Returns the index of the cluster in 'clusters'
# into which would be best to merge new_cluster
def bestMergeIndex(clusters, new_cluster):

    max_quality_merge = 0 
    max_quality_merge_index = 0

    for i in range(len(clusters)):
        c = clusters[i]
        quality = qualityOfMerge(c, new_cluster)

        if quality > max_quality_merge:
            max_quality_merge = quality
            max_quality_merge_index = i

    return clusters[max_quality_merge_index]


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

vocab_size = 2000 #len(sorted_word_frequencies)
 
# Create starting clusters for each of the first 1000 most-common words
clusters = []
num_clusters = 1000
for i in range(num_clusters):
    c = Cluster(sorted_word_frequencies[i][0])
    clusters.append(c)

# Add the rest of the words to the existing clusters
for i in range(num_clusters, vocab_size):

    # Create a new cluster for the next word
    new_cluster = Cluster(sorted_word_frequencies[i][0]) 
    
    # Merge the new cluster into one of the 1000 clusters
    # for which it is most similar
    bestMergeIndex(clusters, new_cluster).merge(new_cluster)

for c in clusters:
    print c.words
