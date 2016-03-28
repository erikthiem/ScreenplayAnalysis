################################################
# GenerateLinesByGenre                         #
# This file generates lines tagged with genre  #
# and stores them in a sqlite database.        #
################################################

import sqlite3
import os

class Movie():
    
    def __init__(self, movie_number, genres):
        self.movie_number = movie_number
        self.genres = genres


class Line():

    def __init__(self, movie_number, text):
        self.movie_number = movie_number
        self.text = text
        self.genre = ""


# File information
lines_file_name = 'movie_lines.txt'
movie_metadata_file_name = 'movie_titles_metadata.txt'
input_files_relative_path = '../raw/'

database_file_name = 'lines_by_genre.db'
output_files_relative_path = '../processed/'

# Error checking for database file existence
database_filename = output_files_relative_path + database_file_name
if os.path.isfile(database_filename):
    answer = raw_input("\nDatabase '{0}' already exists. Are you sure you want to delete and re-create it? (y/n): ".format(database_filename)).lower()

    if (answer == 'y' or answer == 'yes'):
        print("Re-creating database.\n")
        os.remove(database_filename)
    else:   
        print("Exiting.\n")
        exit(-1)
 


# Load movie metadata
with open(input_files_relative_path + movie_metadata_file_name) as f:
    movie_metadata = f.readlines()

# Process and save movie metadata
movie_objects = []
for movie in movie_metadata:
    parts = movie.strip().split("+++$+++")
    movie_number = int(parts[0].replace("m", ""))
    genres = parts[5][1:].split("[")[1].split("]")[0].replace(", ","").replace("\'\'", ",").replace("\'","")
    m = Movie(movie_number, genres)
    movie_objects.append(m)


# Load lines
with open(input_files_relative_path + lines_file_name) as f:
    lines = f.readlines()

# Process and save lines
line_objects = []
for line in lines:
    parts = line.strip().split(" +++$+++ ")
    movie_number = int(parts[2].replace("m", ""))
    text = parts[-1]
    l = Line(movie_number, text)
    line_objects.append(l)

# Generate dictionary to map movie number to genres
number_to_genres = {}
for movie in movie_objects:
    number_to_genres[movie.movie_number] = movie.genres

# Create the SQL table
conn = sqlite3.connect(output_files_relative_path + database_file_name)
c = conn.cursor()
c.execute("CREATE TABLE lines (genres text, line text)")

# Insert the lines
for line in line_objects:
    text = line.text.decode("ascii", "ignore")
    genres = number_to_genres[line.movie_number]
    c.execute("INSERT INTO lines VALUES (?, ?)", (genres, text))

conn.commit()
conn.close()
