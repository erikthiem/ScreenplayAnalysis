#!/usr/bin/python

import sys, sqlite3, re, string

class train:
	
	### Initialize ###
	
	# Files
	wordFile = open('wordFile.txt','w+')
	countFile = open('countFile.txt','w+')
	percentFile = open('percentFile.txt','w+')

	# Lists of genres
	genreList = ['comedy', 'romance', 'adventure', 'biography', 'drama', 'history', 'action', 'crime', 'thriller', 'mystery', 'sci-fi', 'fantasy', 'horror', 'music', 'western', 'war', 'adult', 'musical', 'animation', 'sport', '', 'family', 'short', 'film-noir', 'documentary']
	genreWords = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	genreCounts = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	genrePercents = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
	totalWords = 0.0

	# Read in sentences from train.db
	conn = sqlite3.connect('../../data/processed/train.db')
	print "Successful connection to training database"
	dbCursor = conn.execute("SELECT * FROM lines")

	# Get sentence's genre
	i = 0
	for row in dbCursor:
		i += 1
		print float(i)/274242.0
		genres = row[0].split(',')
		sentence = [x.lower() for x in row[1].split()]

		# Count number of times word is used in each genre
		for word in sentence:
			totalWords += 1.0

			# If word hasn't appeared yet, add it to list
			for genre in genres:
				index = genreList.index(genre)
				if genreWords[index].count(word) == 0:
					genreWords[index].append(word)
					genreCounts[index].append(1)
			
			# Otherwise increment its count by 1
				else:
					genreCounts[index][genreWords[index].index(word)] += 1

	# Write results to file
	wordFile.write(str(genreWords))
	countFile.write(str(genreCounts))
	
	print "Training Finished"

	# Close connection to databases
	conn.close()
	wordFile.close()
	countFile.close()
	percentFile.close()
