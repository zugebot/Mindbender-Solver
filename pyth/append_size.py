import os

# Specify the directory you want to work with


# Iterate over each file in the directory
for filename in os.listdir():
    # Check if the file is a .txt file
    if filename.endswith('.txt'):
        # Construct the full file path
        filepath = filename
        if filename.count("_") != 1:
            continue

        # Count the number of lines in the file
        with open(filepath, 'r') as file:
            lines = file.readlines()
            num_lines = len(lines)

        # Create the new file name with _{LEN} before .txt
        name_part, ext = os.path.splitext(filename)
        new_filename = f"{name_part}_{num_lines}{ext}"

        # Construct the new file path
        new_filepath = new_filename

        # Rename the file
        os.rename(filepath, new_filepath)

        print(f"Renamed: {filename} -> {new_filename}")
