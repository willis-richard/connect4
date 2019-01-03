"""These classes are not currently used as a dict() is faster"""

from sortedcontainers import SortedSet


class Entry():
    def __init__(self, board, node_data):
        self.age = board.age
        self.table = dict()
        self.table[board] = node_data

    def __eq__(self, obj):
        return self.age == obj

    def __gt__(self, other):
        return self.age > other

    def __lt__(self, other):
        return self.age < other

    def __hash__(self):
        return hash(self.age)

    def __repr__(self):
        return str(self.age) + "\n" + str(self.table)


class TransitionTable():
    def __init__(self):
        # sorted board evaluations by the number of pieces
        self.entries = SortedSet()

    def __contains__(self, board):
        age = board.age
        if age in self.entries:
            idx = self.entries.bisect_left(age)
            return board in self.entries[idx].table

        return False

    def __getitem__(self, board):
        age = board.age
        if age in self.entries:
            idx = self.entries.bisect_left(age)
            return self.entries[idx].table[board]

        raise KeyError("item not in TransitionTable")

    def __setitem__(self, board, node_data):
        age = board.age
        if age in self.entries:
            idx = self.entries.bisect_left(age)
            self.entries[idx].table[board] = node_data
        else:
            self.entries.add(Entry(board, node_data))

    def age(self, num_moves):
        idx = self.entries.bisect_left(num_moves - 1)
        if idx > 0:
            for _ in range(idx):
                del self.entries[0]

    def __repr__(self):
        return repr(self.entries)


class TransitionTableDictOfDict():
    def __init__(self):
        # sorted board evaluations by the number of pieces
        self.entries = dict()

    def __contains__(self, board):
        age = board.age
        if age in self.entries:
            return board in self.entries[age]

        return False

    def __getitem__(self, board):
        age = board.age
        if age in self.entries:
            return self.entries[age][board]

        raise KeyError("item not in TransitionTable")

    def __setitem__(self, board, node_data):
        age = board.age
        if age in self.entries:
            self.entries[age][board] = node_data
        else:
            self.entries[age] = dict()
            self.entries[age][board] = node_data

    def age(self, num_moves):
        return

    def __repr__(self):
        return repr(self.entries)
