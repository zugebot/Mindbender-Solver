import os
import re

# Iterate over each file in the current directory
for filename in os.listdir('../levels'):
    # Check if the file is a .txt file and contains a pattern like _{LEN} before .txt
    if filename.endswith('.txt'):
        # Use a regular expression to match and remove the last _{LEN} part
        new_filename = re.sub(r'(_\d+)(?=\.txt$)', '', filename)

        # Only rename if there's a change to avoid unnecessary operations
        if new_filename != filename:
            os.rename(filename, new_filename)
            print(f"Renamed: {filename} -> {new_filename}")
