import sys
import sqlite3
import os
import numpy
from scipy.sparse import lil_matrix
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class Sentence():
    
    def __init__(self, genre, text):
        self.genre = genre
        self.text = text


def sentencesFromDB(db_file_path):

    conn = sqlite3.connect(db_file_path)
    c = conn.cursor()
    c.execute('SELECT * FROM lines;')
    results = c.fetchall()
    conn.commit()
    conn.close() 

    sentences = [] 
    for r in results:
        s = Sentence(r[0], r[1])
        sentences.append(s)

    return sentences


# Generate "X[sentence_id, word_id] = word_count" sparse matrix
def generateWordCountMatrixTrain(training_sentences, word_to_id, id_to_word):

    sentence_word_frequencies = []

    # Create a dictionary to store all words in order to get vocabulary size
    all_word_dict = defaultdict(int)

    for sentence in training_sentences:
        words_with_stopwords = word_tokenize(sentence.text.lower())

        words = []
        english_stopwords = stopwords.words('english')
        for word in words_with_stopwords:
            if word not in english_stopwords:
                words.append(word)

        word_frequencies = {w:words.count(w) for w in set(words)}
        sentence_word_frequencies.append(word_frequencies)
        for word in words:
            all_word_dict[word] += 1

    # Create dictionaries to do "word" -> "word ID" and "word ID" -> "word"
    word_id = 0
    for word in all_word_dict.iteritems():
        word_to_id[word[0]] = word_id
        id_to_word[word_id] = word[0]
        word_id += 1

    # Create blank X matrix
    X = lil_matrix( (len(training_sentences), len(all_word_dict)))

    # Populate X matrix with the values from the word frequencies in each sentence
    for sentence in range(len(sentence_word_frequencies)):
        for word, frequency in sentence_word_frequencies[sentence].iteritems():
            X[sentence, word_to_id[word]] = frequency

    # Convert X to sparse CSR format
    X = X.tocsr()

    return X

# Generate "X[sentence_id, word_id] = word_count" sparse matrix
def generateWordCountMatrixTest(testing_sentences, trainX, word_to_id, id_to_word):

    sentence_word_frequencies = []

    for sentence in testing_sentences:
        words = word_tokenize(sentence.text.lower())
        word_frequencies = {w:words.count(w) for w in set(words)}
        sentence_word_frequencies.append(word_frequencies)


    # Create blank X matrix
    X = lil_matrix( (len(testing_sentences), trainX.shape[1]))

    # Populate X matrix with the values from the word frequencies in each sentence
    for sentence in range(len(sentence_word_frequencies)):
        for word, frequency in sentence_word_frequencies[sentence].iteritems():
            if word in word_to_id:
                X[sentence, word_to_id[word]] = frequency

    # Convert X to sparse CSR format
    X = X.tocsr()

    return X



# Generate "Y[sentence_id] = genre" vector
def generateGenreVector(training_sentences):

    Y = numpy.zeros((len(training_sentences), 1), dtype=int)
    
    for sentence in range(len(training_sentences)):
        if "comedy" in training_sentences[sentence].genre:
            Y[sentence] = 1
        else:
            Y[sentence] = -1

    return Y


def train(X, Y, num_iterations, learning_rate):

    weights = numpy.zeros(X.shape[1], dtype=float)

    for iteration in range(num_iterations):
        
        print "Starting iteration #{0}".format(iteration)
        print "{0}".format(weights[0:20])

        for sentence in range(X.shape[0]):

            word_counts_in_sentence = numpy.squeeze(numpy.asarray(X[sentence].todense()))
            dot_product = word_counts_in_sentence.dot(weights)

            if numpy.sign(dot_product) != Y[sentence]:
                weights += learning_rate*( (Y[sentence] - numpy.sign(dot_product)) * (word_counts_in_sentence) ) 

    return weights


def predict(X, weights):

    predictions = numpy.zeros(X.shape[0], dtype=int) 

    for sentence in range(X.shape[0]):

        row_as_array = numpy.squeeze(numpy.asarray(X[sentence].todense()))
            
        predictions[sentence] = numpy.sign(row_as_array.dot(weights))

    return predictions


def percentSimilar(list1, list2):

    total = len(list1)
    count_similar = 0

    for i in range(len(list1)):
        if list1[i].item() == list2[i]:
            count_similar += 1

    return float(count_similar) / total


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print "\nError! Expected syntax: 'python perceptron.py training_db_path testing_db_path num_iterations'\n"
        sys.exit(-1)

    training_data_path = sys.argv[1]
    testing_data_path = sys.argv[2]
    num_iterations = int(sys.argv[3])
    learning_rate = float(sys.argv[4])

    if not os.path.isfile(training_data_path):
        print "Error! Training database {0} does not exist.\n".format(training_data_path)
        exit(-1)

    if not os.path.isfile(testing_data_path):
        print "Error! Testing database {0} does not exist.\n".format(testing_data_path)
        exit(-1)

    # Load the Training data
    training_sentences = sentencesFromDB(training_data_path)

    # Generate "X[sentence_id, word_id] = word_count" sparse matrix
    word_to_id = {}
    id_to_word = {}
    trainX = generateWordCountMatrixTrain(training_sentences, word_to_id, id_to_word)

    # Generate "Y[sentence_id] = genre" vector
    trainY = generateGenreVector(training_sentences)

    weights = train(trainX, trainY, num_iterations, learning_rate)

    # Load the testing data
    testing_sentences = sentencesFromDB(testing_data_path)

    testX = generateWordCountMatrixTest(testing_sentences, trainX, word_to_id, id_to_word)  

    # Predict each of the testing sentences
    predicted_classifications = predict(testX, weights)

    # Determine accuracy of predictions
    correct_classifications_genre = [s.genre for s in testing_sentences]
    correct_classifications = []

    for c in correct_classifications_genre:
        if c == "Democratic":
            correct_classifications.append(1)
        else:
            correct_classifications.append(-1)

    accuracy = percentSimilar(predicted_classifications, correct_classifications)

    frequency = [(index, count) for index,count in enumerate(weights)]
    frequency.sort(key=lambda tup: tup[1], reverse=True)
    print("\nMost Comedic Words:")
    for i in range(20):
        print("{0} : {1}".format(id_to_word[frequency[i][0]], frequency[i][1]))
    frequency.sort(key=lambda tup: tup[1])
    print("\nLeast Comedic Words:")
    for i in range(20):
        print("{0} : {1}".format(id_to_word[frequency[i][0]], frequency[i][1]))

    print("Accuracy: {0}".format(accuracy))
