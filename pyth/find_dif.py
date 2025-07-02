
name = "5-3"
size = 7

with open(f"{name}_random.txt", "r") as f1:
    lines1 = f1.readlines()
    lines1 = [line.strip("\n").strip(" ") for line in lines1 if line != ""]


all_lines = set()
for line in lines1:
    all_lines.add(line)


with open(f"{name}_c{size}_{len(all_lines)}.txt", "w") as f:
    f.write("\n".join(all_lines))


for line in all_lines:
    print(line)
print(f"\nAll Solutions: {len(all_lines)}")
