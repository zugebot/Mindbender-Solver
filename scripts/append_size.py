import os


for filename in os.listdir():
    if filename.endswith('.txt'):
        filepath = filename
        if filename.count("_") != 1:
            continue

        with open(filepath, 'r') as file:
            lines = file.readlines()
            num_lines = len(lines)

        name_part, ext = os.path.splitext(filename)
        new_filename = f"{name_part}_{num_lines}{ext}"

        new_filepath = new_filename

        os.rename(filepath, new_filepath)

        print(f"Renamed: {filename} -> {new_filename}")
