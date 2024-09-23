echo Loading .gdbinit\n
python
import sys
# Update the path below to match the actual path
sys.path.insert(0, 'C:/Users/jerrin/CLionProjects/Mindbender-Solver/MindbenderSolver/pretty-printers')
import memory_printer
# memory_printer.register_memory_printers(gdb.current_objfile())
end
