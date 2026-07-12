

import re


with open("4-4_c8.txt", "r") as f:
    text = f.read()

    pattern = r"[RC][0145][0-6] [RC][0145][0-6] [RC][0145][0-6] [RC][0145][0-6] " \
              r"[RC][0145][0-6] [RC][0145][0-6] [RC][0145][0-6] [RC][0145][0-6] "

    items = re.findall(pattern, text)

    for n, i in enumerate(items):
        print(f"Solution #{n+1}:", i)














