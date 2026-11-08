import re
import itertools


def get_permutations(s):
    moves = s.split()

    # Find all contiguous blocks of R or C moves
    groups = []
    current_group = [moves[0]]

    for move in moves[1:]:
        if move[0] == current_group[-1][0]:
            current_group.append(move)
        else:
            groups.append(current_group)
            current_group = [move]

    groups.append(current_group)

    all_combinations = []
    for group in groups:
        if len(group) > 1:
            all_combinations.append(list(itertools.permutations(group)))
        else:
            all_combinations.append([group])

    all_orders = list(itertools.product(*all_combinations))

    result = [' '.join(' '.join(group) for group in order) for order in all_orders]

    return result
