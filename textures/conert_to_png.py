




import os


files = os.listdir()

print(files)

for i in files:
    if i.endswith(".gif"):
        os.rename(i, i.replace(".gif", ".png"))
    if i.endswith(".jpg"):
        os.rename(i, i.replace(".jpg", ".png"))










