import win32gui
from PIL import ImageGrab
from math import *
import pyautogui
import time

def permutation(direction, sect, amount, g):
    a1=g[0];a2=g[1];a3=g[2];a4=g[3];a5=g[4];a6=g[5];
    b1=g[6];b2=g[7];b3=g[8];b4=g[9];b5=g[10];b6=g[11];
    c1=g[12];c2=g[13];c3=g[14];c4=g[15];c5=g[16];c6=g[17];
    d1=g[18];d2=g[19];d3=g[20];d4=g[21];d5=g[22];d6=g[23];
    e1=g[24];e2=g[25];e3=g[26];e4=g[27];e5=g[28];e6=g[29];
    f1=g[30];f2=g[31];f3=g[32];f4=g[33];f5=g[34];f6=g[35];

    if direction == 0: #Vertical
        for i in range(amount - 1):
            p = locals()["a"+X123[sect-1:sect]] #column
            if sect == 1:
                p1=f1;f1=e1;e1=d1;d1=c1;c1=b1;b1=a1;a1=p1
            elif sect == 2:
                p2=f2;f2=e2;e2=d2;d2=c2;c2=b2;b2=a2;a2=p2
            elif sect == 3:
                p3=f3;f3=e3;e3=d3;d3=c3;c3=b3;b3=a3;a3=p3
            elif sect == 4:
                p4=f4;f4=e4;e4=d4;d4=c4;c4=b4;b4=a4;a4=p4
            elif sect == 5:
                p5=f5;f5=e5;e5=d5;d5=c5;c5=b5;b5=a5;a5=p5
            else:
                p6=f6;f6=e6;e6=d6;d6=c6;c6=b6;b6=a6;a6=p6
    elif direction == 1: #Horizontal
        for i in range(amount - 1):
            p = locals()[Xabc[sect-1:sect] + "1"] #row
            if sect == 1:
                p=a6;a6=a5;a5=a4;a4=a3;a3=a2;a2=a1;a1=p
            elif sect == 2:
                p=b6;b6=b5;b5=b4;b4=b3;b3=b2;b2=b1;b1=p
            elif sect == 3:
                p=c6;c6=c5;c5=c4;c4=c3;c3=c2;c2=c1;c1=p
            elif sect == 4:
                p=d6;d6=d5;d5=d4;d4=d3;d3=d2;d2=d1;d1=p
            elif sect == 5:
                p=e6;e6=e5;e5=e4;e4=e3;e3=e2;e2=e1;e1=p
            else:
                p=f6;f6=f5;f5=f4;f4=f3;f3=f2;f2=f1;f1=p
    return [a1,a2,a3,a4,a5,a6,b1,b2,b3,b4,b5,b6,c1,c2,c3,c4,c5,c6,d1,d2,d3,d4,d5,d6,e1,e2,e3,e4,e5,e6,f1,f2,f3,f4,f5,f6]


def Compare(grid): #grid is a 36-length list,  Find the matches! REEEEEEEEEEEE
    matrix = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for v in range(36):
        pixel,squares = grid[v],tuples[v]
        #matrix[v] is the pinpoint
        #matrix[i-1] are the spots we test around [v]
        numSim = 0
        for i in squares:
            if grid[v] == grid[i-1]:
                numSim = numSim + 1
                #if matrix[i-1]==0:
                #    matrix[v]=matrix[v]+1
                #else:
                #    matrix[v]=matrix[i-1]+1
        matrix[v] = numSim       
    value = [max(matrix),matrix.count(max(matrix)),matrix]
    return value


def disp(l):
    for x in [0,6,12,18,24,30]:
        print(l[x],l[x+1],l[x+2],l[x+3],l[x+4],l[x+5],sep=" ")


Xabc = "abcdef"
X123 = "123456"
xL = [222,286,349,414,477,541]
yL = [104,170,232,300,360,427]
xlist,ylist = [],[]
index = ["200","210","221","020","122","002","202","222","220","010","100","022"]
#colors = ["R","O","Y","G","b","B","P","W","Y","G","R","b"]
colors = ["1","2","3","4","5","6","7","8","3","4","1","5"]
tuples = [(2,7),(1,3,8),(2,4,9),(3,5,10),(4,6,11),(5,12),(1,8,13),(2,7,9,14),(3,8,10,15),(4,9,11,16),(5,10,12,17),(6,11,18),(7,14,19),(8,13,15,20),(9,14,16,21),(10,15,17,22),(11,16,18,23),(12,17,24),(13,20,25),(14,19,21,26),(15,20,22,27),(16,21,23,28),(17,22,24,29),(18,23,30),(19,26,31),(20,25,27,32),(21,26,28,33),(22,27,29,34),(23,28,30,35),(24,29,36),(25,32),(26,31,33),(27,32,34),(28,33,35),(29,34,36),(30,35)]



def iter(grid):       #Find the most matches
    listc = []
    matchbig,matchnum,best = 0,0,0
    for x in range(60):
        q,a,r = int(x/30), int(x/5)%6+1, x%5+1
        current = Compare(permutation(q,r,a+1,grid))
        if matchbig < current[0]:
            disp(current[2])
            print("")
            best = current
            matchbig,matchnum = current[0],current[1]
            print("{} sets of matching {}'s.".format(matchnum,matchbig))
        #import sys;sys.exit(0)
    print(current)
    return best


def callback(hwnd, extra):
    global xlist,ylist,value
    r = win32gui.GetWindowRect(hwnd)
    x,y = r[0],r[1]
    w = r[2] - x
    h = r[3] - y
    xlist = tuple(map(lambda k: k+5, xL))
    ylist = tuple(map(lambda k: k+0, yL))
    alist = []

    if win32gui.GetWindowText(hwnd) == "Chuzzle Deluxe 1.01":
        print("%s" % win32gui.GetWindowText(hwnd))
        print("\tLocation: (%d, %d)" % (x, y))
        print("\t    Size: (%d, %d)" % (w, h))
        TS = time.time()
        time.sleep(2)
        image = ImageGrab.grab()
        for yy in range(6):
            seg = ""
            for xx in range(6):
                color = image.getpixel((xlist[xx]+x,ylist[yy]+y))
                a = tuple(map(lambda x: str(floor(x/95)), color))
                a = a[0]+a[1]+a[2]
                for i in range(12):
                    a = a.replace(index[i],colors[i])
                seg = seg + a
                alist = alist + [a]
            print(seg)
        value = alist
        print(time.time()-TS)
        

def main():
    win32gui.EnumWindows(callback, None)
    var = iter(value)
    print(var)
    disp(var[2])

if __name__ == '__main__':
    main()

print(pyautogui.position())





