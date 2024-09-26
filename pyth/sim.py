import math
from copy import deepcopy

GRID_SIZE = 6


def calculate_distance(p1, p2):
    """Calculate Euclidean distance without wrap-around for positions possibly outside the grid."""
    if p1 is None or p2 is None:
        return 0
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    return math.hypot(dx, dy)


def get_positions_on_axis(axis, index):
    """Get all positions along the given axis (row or column index)."""
    if axis == 'R':
        return [[x, index] for x in range(GRID_SIZE)]
    else:
        return [[index, y] for y in range(GRID_SIZE)]


def get_possible_dirs_and_displacements(steps):
    """Get possible directions and corresponding displacements."""
    steps = steps % GRID_SIZE
    if steps == 0:
        return [('none', 0)]
    else:
        return [
            ('+', steps),
            ('-', -((GRID_SIZE - steps) % GRID_SIZE))
        ]


def optimize_mouse_movement_pass1(solution_moves):

    # Initialize variables
    move_data = [None] * len(solution_moves)
    next_click_down_pos = None

    for move_index in range(len(solution_moves) - 1, -1, -1):
        current_move = solution_moves[move_index]
        parsed_move = [current_move[0], int(current_move[1]), int(current_move[-1])]

        axis = parsed_move[0] != 'R'

        best_option = None
        min_total_distance = float('inf')

        # Iterate over all possible click-down positions
        for click_down_pos in get_positions_on_axis(parsed_move[0], parsed_move[1]):
            # Iterate over possible directions
            for direction, displacement in get_possible_dirs_and_displacements(parsed_move[2]):
                # Calculate release position
                release_pos = [click_down_pos[0], click_down_pos[1]]
                release_pos[axis] += displacement

                if move_index != 0 or next_click_down_pos is not None:
                    release_pos[0] %= GRID_SIZE
                    release_pos[1] %= GRID_SIZE

                # Calculate distances
                distance_to_next = calculate_distance(release_pos, next_click_down_pos)
                drag_distance = calculate_distance(click_down_pos, release_pos)

                # Since we're working backwards,
                # we don't need to calculate distance from current position
                total_distance = drag_distance + distance_to_next

                # Choose the best option
                if total_distance < min_total_distance:
                    min_total_distance = total_distance
                    best_option = {
                        'move': current_move,
                        'click_down': click_down_pos,
                        'release': release_pos,
                        'direction': direction,
                        'displacement': displacement,
                        'total_distance': total_distance
                    }

        # update our moves with the best option
        move_data[move_index] = best_option
        next_click_down_pos = best_option['click_down']

    return move_data


"""
going from beginning to end:


R45 C11 C24 R35 R41 C41 C53 R14 R21

find all moves that go in the same direction.
1: R (keep looking for R's)
2: C (done!)
look through all combinations of [1] and find the one that minimizes the distance of ending on column [2]
NEXT

2: C (keep looking for C's)
3: C (keep looking for C's)
4: R (done!)
look through all combinations of [2][3] and find the one that minimizes the distance of ending on row [4]
NEXT

4: R (keep looking for R's)
5: R (keep looking for R's)
6: C (done!)
look through all combinations of [4][5] and find the one that minimizes the distance of ending on column [6]
NEXT

6: C (keep looking for C's)
7: C (keep looking for C's)
8: R (done!)
look through all combinations of [6][7] and find the one that minimizes the distance of ending on column [8]
NEXT

8: R (keep looking for R's)
it is the last move, so (done!)
"""





def optimize_mouse_movement_pass2(move_data):
    if len(move_data) < 2:
        return

    neg1_move = move_data[-1]
    neg2_move = move_data[-2]

    neg1_pos = neg1_move['click_down']
    neg2_pos = neg2_move['release']
    if neg1_move['move'][0] == neg2_move['move'][0]:
        return

    pos_x_same = neg1_pos[0] == neg2_pos[0]
    pos_y_same = neg1_pos[1] == neg2_pos[1]
    if pos_x_same ^ pos_y_same:

        if pos_x_same:  # y values are different
            y_dif = neg2_pos[1] - neg1_pos[1]
            move_data[-1]['click_down'][1] += y_dif
            move_data[-1]['release'][1] += y_dif

        elif pos_y_same:  # x values are different
            x_dif = neg2_pos[0] - neg1_pos[0]
            move_data[-1]['click_down'][0] += x_dif
            move_data[-1]['release'][0] += x_dif


def optimize_mouse_movement_pass3(move_data):
    i = 0
    while i < len(move_data) - 2:
        current = move_data[i]
        next1 = move_data[i + 1]
        next2 = move_data[i + 2]

        if current['move'][0] != next1['move'][0]:
            if current['release'] == next1['click_down']:
                i += 1
                continue

            if next1['release'] != next2['click_down']:

                # if not displaced current next1:
                index = current['move'][0] == "C"
                if current['release'][index] == next1['click_down'][index]:
                    if current['release'][1 - index] != next1['click_down'][1 - index]:
                        shift_amount = current['release'][1 - index] - next1['click_down'][1 - index]
                        next1['click_down'][1 - index] += shift_amount
                        next1['release'][1 - index] += shift_amount

                        distance = next1['click_down'][1 - index] - next1['release'][1 - index]
                        if next1['release'][1 - index] < 0:
                            next1['release'][1 - index] += 6
                        elif next1['release'][1 - index] > 5:
                            next1['release'][1 - index] -= 6
        i += 1


def shift_last_moves(move_data):
    """
    make a list of all moves that follow the same either R or C
    if the release spot between the vice R/C and the first R/C is the same in the x/y,
    we should be able to shift all the cells down/left/up/right maybe?
    """
    count = 0
    last_rc = move_data[-1]['move'][0]
    Rx_or_Cy = last_rc == "C"
    for move in reversed(move_data):
        if move['move'][0] == last_rc:
            count += 1
        else:
            break
    else:
        return

    pass

    bounds = [1000, -1000]
    for i in range(count):
        click_down = move_data[-i - 1]['click_down'][Rx_or_Cy]
        if click_down < bounds[0]:
            bounds[0] = click_down
        if click_down > bounds[1]:
            bounds[1] = click_down

    # figure out the displacement between the first last R/C and the move before it
    displacement = move_data[-count]['click_down'][Rx_or_Cy] - move_data[-count - 1]['release'][Rx_or_Cy]
    lower = bounds[0] - displacement
    upper = bounds[1] - displacement
    if displacement > 0:
        if lower < 0:
            displacement += lower
    else:
        pass
        # if upper > 6:
        #     displacement -= upper
        # pass  # idk how to code this one yet

    if displacement != 0:
        for i in range(count):
            move_data[-i - 1]['click_down'][Rx_or_Cy] -= displacement
            move_data[-i - 1]['release'][Rx_or_Cy] -= displacement
        move_data[-count - 1]['displacement'] -= displacement
        move_data[-count - 1]['total_distance'] -= displacement
    pass


def calculate_total_distance(move_data):
    total_distance_to_drag = 0
    total_distance_to_move = 0
    current_pos = move_data[0]['click_down']
    for move_info in move_data:
        distance_to_move = calculate_distance(current_pos, move_info['click_down'])
        distance_to_drag = calculate_distance(move_info['click_down'], move_info['release'])

        move_info['displacement'] = distance_to_move
        move_info['total_distance'] -= distance_to_drag
        total_distance_to_drag += distance_to_drag
        total_distance_to_move += distance_to_move
        current_pos = move_info['release']

    return total_distance_to_drag, total_distance_to_move


def find_shortest_mouse_path(move_string, state1=True, state2=True, state3=False):
    move_string = move_string.strip()
    if len(move_string) != 3:
        move_list = move_string.split(" ")
    else:
        move_list = [move_string]

    move_data = optimize_mouse_movement_pass1(move_list)

    if state3:
        move_data_copy = deepcopy(move_data)
        optimize_mouse_movement_pass2(move_data)
        if move_data != move_data_copy:
            print("found a difference")

    if state1:
        optimize_mouse_movement_pass3(move_data)

    # shift_last_moves(move_data)
    to_drag, to_move = calculate_total_distance(move_data)
    return to_drag, to_move, move_data
