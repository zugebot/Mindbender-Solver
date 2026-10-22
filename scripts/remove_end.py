import os
import re

for filename in os.listdir('../build/levels'):
    if filename.endswith('.txt'):
        new_filename = re.sub(r'(_\d+)(?=\.txt$)', '', filename)

        if new_filename != filename:
            os.rename(filename, new_filename)
            print(f"Renamed: {filename} -> {new_filename}")
