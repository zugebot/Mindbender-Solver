


from DataManager import Data

file = r"C:\Users\jerrin\Documents\my games\Chuzzle\Profiles\Jerrin\SAVEGAME-SPEEDCHUZZLE.DAT"


data = Data(endian="little")
data.openFile(file)


print(data.nextInt(), "header")
print(data.nextInt(), "gamemode?")
print(data.nextInt(), "gamemode?")
print(data.nextShort())
print(data.nextInt(), "score")
print(data.nextShort(), "scrambles")
print(data.nextInt(), "bottle current")
print(data.nextInt(), "bottle cap")

print("do not know<")
print(data.nextInt()) # ?
print(data.nextInt()) # ?
print(data.nextInt()) # empty
print(data.nextShort())
print(data.nextInt())
print(data.nextInt())
print(data.nextInt())
print(data.nextInt())
print(data.nextInt())
print(">do not know")


print(data.nextShort()) # ?
print(data.nextShort(), "speed bar")



data.index = len(data.data) - 1512

print("chuzz")

chuz = []
for i in range(36):
    items = []
    items += [data.nextShort()] # color


    items += [data.nextBytes(36)]

    items += [data.nextInt()]
    chuz.append(items)

for n, i in enumerate(chuz):
    print(n, i)








data.saveFile(file)











