#!/usr/bin/python

import sys


class learn:

	### Initialization ###

	# Open text files
	with open("republicanWords.txt","r+") as repFile:
		repFileContent = repFile.readlines()
	with open("democratWords.txt","r+") as demFile:
		demFileContent = demFile.readlines()
	with open("words.txt","r+") as wordsFile:
		wordsFileContent = wordsFile.readlines()

	# Create files of P(word|party)
	repPer = open("republicanPercent.txt","wr+")
	demPer = open("democratPercent.txt","wr+")

	# Variables
	repVocabSize = 0
	demVocabSize = 0
	vocabSize = 0
	pWord = 0.0

	### Learning ###

	# Calculate number of terms - Republican
	for line in repFileContent:
		repVocabSize = repVocabSize + 1
		#line = line.split("\t")
		#repTotal = repTotal + int(line[1])

	# Calculate number of terms - Democrat
	for line in demFileContent:
		demVocabSize = demVocabSize + 1
		#line = line.split("\t")
		#demTotal = demTotal + int(line[1])

	# Calculate total number of terms
	for line in wordsFileContent:
		vocabSize = vocabSize + 1

	# Calculate P(word|Republican) w/ Laplace Smoothing
	for line in repFileContent:
		line = line.split("\t")
		pWord = float((float(line[1]) + 1) / (repVocabSize + vocabSize))
		repPer.write(line[0] + "\t" + str(pWord) + "\n")

	# Calculate P(word|Democrat) w/ Laplace Smoothing
	for line in demFileContent:
		line = line.split("\t")
		pWord = float((float(line[1]) + 1) / (demVocabSize + vocabSize))
		demPer.write(line[0] + "\t" + str(pWord) + "\n")

	# Close files
	repFile.close()
	demFile.close()
	wordsFile.close()
	repPer.close()
	demPer.close()
