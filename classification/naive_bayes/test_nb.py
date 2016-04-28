#!/usr/bin/python

import sys, sqlite3, os, time
from nb_Classifier import genrePercents

class test:

	# Read in sentences from test.db
	connTest = sqlite3.connect('../../data/processed/test.db')
	print "Successful connection to training database"
	dbCursorTest = connTest.execute("SELECT * FROM lines")
	
	# Predict sentence genre
	j = 0
	selected = ""
	correct = 0
	wrong = 0.0
	for row in dbCursorTest:
		i += 1
		print float(i)/27424.0
		genresTest = row[0].split(',')
		sentenceTest = [x.lower() for x in row[1].split()]
		testResults = []

		# Calculate number of genres assigned to sentence
		numGenres = len(genresTest)
		
		# Calculate percent for each genre
		for genre in genreList:
			rowSum = 0.0
			for word in sentence:
				rowSum += genrePercents[genre][word]
			testResults.append(rowSum)

		# Determine genres for given sentence
		k = 0
		while k < numGenres:
			selected = genreList.index(max(testResults))
			if genresTest.count(selected) == 0:
				wrong += 1.0
			else:
				correct += 1
			k += 1

	# Calculate accuracy
	accuracy = float(correct)/wrong

	print "Accuracy: ",accuracy

	print "Time: ",time.clock()

