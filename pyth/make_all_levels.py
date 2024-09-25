import os
from permutations import get_permutations

files = os.listdir("levels")
files = [file for file in files if file.endswith(".txt")]

for file in files:

    with open(f"levels/{file}", "r") as fr:
        lines = [line.strip("\n") for line in fr.readlines() if line != ""]

    _solutions = set()
    for line in lines:
        _temp = get_permutations(line)
        for solution in _temp:
            _solutions.add(solution)
    all_solutions = list(_solutions)
    all_solutions = sorted(all_solutions)

    level_name, move_count, _ = file.split("_")
    new_filename = f"{level_name}_{move_count}_{len(all_solutions)}.txt"

    with open(f"all_levels/{new_filename}", "w") as fw:
        fw.write("\n".join(all_solutions))
