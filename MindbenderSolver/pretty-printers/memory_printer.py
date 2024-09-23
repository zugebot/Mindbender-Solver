import gdb
import re

class MemoryPrinter:
    "Pretty-printer for the Memory class"

    def __init__(self, val):
        self.val = val

    def to_string(self):
        # Custom summary string similar to the C++ toString method
        moves = self.val['moves']
        move_count = moves & 0xF
        move_list = []
        for i in range(move_count):
            shift_amount = 4 + (i * 6)
            move = (moves >> shift_amount) & 0x3F
            move_list.append(move)

        results = []
        for move in move_list:
            direction = move / 30
            sect = (move % 30) / 5
            amount = move % 5 + 1
            results.append("RC"[direction] + str(sect) + str(amount))
        result = ", ".join(results)
        return result

    def children(self):
        # Return the fields of the Memory class to allow them to be expanded
        yield "moves", self.val['moves']  # The 'moves' field will be collapsable

    def display_hint(self):
        # This tells GDB how to display the root item. 'map' is used for expandable objects.
        return 'map'


def memory_lookup_function(val):
    if re.search(r'^Memory$', str(val.type)):
        return MemoryPrinter(val)
    return None


def register_memory_printers(objfile):
    gdb.printing.register_pretty_printer(objfile, memory_lookup_function)


if __name__ == "__main__":
    gdb.current_objfile().pretty_printers.append(register_memory_printers)
