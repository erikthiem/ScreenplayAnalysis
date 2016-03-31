################################################
# An implementation of the Brown clustering    #
# algorithm.                                   #
# This file outputs each word with its         #
# bitstring, sorted by bitstring.              #
################################################

import sqlite3
import os


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
all_results = c.fetchall()
conn.commit()
conn.close()

lines = [entry[1] for entry in all_results]

