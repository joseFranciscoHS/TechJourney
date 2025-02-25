import ctypes


class DynamicArray:
    """Dynamic array implementation"""

    def __init__(self, default_capacity: int = 1):
        self.n = 0
        self.capacity = default_capacity
        self.array = self.make_array(len=self.capacity)

    def make_array(self, len: int):
        "generate a new array of len capacity"
        return (len * ctypes.py_object)()

    def __len__(self):
        return self.n

    def __getitem__(self, i: int):
        if -1 < i <= self.n:
            return self.array[i]

        raise IndexError("Index out of bounds")

    def resize(self):
        "generates a new array with twice capacity"
        new_capacity = self.capacity * 2
        tmp_array = self.make_array(new_capacity)

        for i in range(self.n):
            tmp_array[i] = self.array[i]

        self.array = tmp_array
        self.capacity = new_capacity

    def append(self, value):
        "add a new value at the end of the array"
        if self.n == self.capacity:
            self.resize()

        self.array[self.n] = value
        self.n += 1

    def delete(self):
        "removes the last element"

        if self.n == 0:
            raise IndexError("Array is empty")

        self.array[self.n - 1] = None
        self.n -= 1

    def insert_at(self, index: int, value):

        if index > self.n or index < 0:
            raise IndexError("Bad index")

        if self.n == self.capacity:
            self.resize()

        for i in range(self.n - 1, index - 1, -1):
            self.array[i + 1] = self.array[i]

        self.array[index] = value
        self.n += 1

    def drop(self, index: int):
        "removes value at position index"

        if not self.n or index < 0 or index > self.n:
            raise IndexError("Index out of bounds")

        for i in range(index, self.n - 1):
            self.array[i] = self.array[i + 1]

        self.array[self.n - 1] = None
        self.n -= 1

    def __str__(self):
        return f"[{', '.join(str(self.array[i]) for i in range(self.n))}]"


if __name__ == "__main__":

    A = DynamicArray()
    A.append(0)
    A.append(1)
    A.append(2)
    A.append(3)
    print(A)

    A.append(4)
    A.append(5)
    A.drop(4)
    A.drop(4)
    print(A)

    A.append(4)
    A.append(6)
    A.insert_at(5, 5)
    print(A)

    A.delete()
    print(A)
