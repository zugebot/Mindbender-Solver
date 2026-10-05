# Jerrin Shirks

class Data:
    def __init__(self, data=b'', endian="little"):
        self.index = 0
        self.data = data
        self.endian = endian

    def toInt(self, val):
        return int.from_bytes(val, self.endian)

    def toChr(self, val):
        if type(val) == int:
            return chr(val)
        else:
            return chr(int.from_bytes(val, self.endian))

    # creating instances of this object
    def openFile(self, filename):
        self.index = 0
        with open(filename, "rb") as f:
            self.data = b''
            while byte := f.read(1):
                self.data += byte

    def saveFile(self, filename):
        with open(filename, "wb") as file:
            file.write(bytearray(self.data))

    # this is for reading from a file
    def nextBytes(self, length):
        sect = self.data[self.index:self.index + length]
        self.index += length
        return sect

    def nextInt(self):
        sect = self.data[self.index:self.index + 4]
        self.index += 4
        return self.toInt(sect)

    def nextShort(self):
        sect = self.data[self.index:self.index + 2]
        self.index += 2
        return self.toInt(sect)

    def nextByte(self):
        sect = self.data[self.index:self.index + 1]
        self.index += 1
        return self.toInt(sect)

    def nextString(self, length):
        sect = self.data[self.index:self.index + length]
        self.index += length
        string = "".join([self.toChr(s) for s in sect])
        return string

    # this is for replacing bytes while iterating over a file
    def replaceBytes(self, value, byteAmount):
        if type(value) == int:
            value = value.to_bytes(byteAmount, self.endian)
        self.data = self.data[:self.index] + value + self.data[self.index + byteAmount:]
        self.index += byteAmount

    def replaceInt(self, value):
        self.replaceBytes(value, 4)

    def replaceShort(self, value):
        self.replaceBytes(value, 2)

    def replaceByte(self, value):
        self.replaceBytes(value, 1)

    # This is for writing to a file

    def writeInt(self, value):
        self.writeBytes(value, 4)

    def writeShort(self, value):
        self.writeBytes(value, 2)

    def writeByte(self, value):
        self.writeBytes(value, 1)

    def writeString(self, value):
        length = len(value)
        self.writeInt(length)
        self.writeBytes(value, length)

    def writeBytes(self, value, length):
        if type(value) == int:
            value = value.to_bytes(length, self.endian)
        self.data += value





