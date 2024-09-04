import numpy as np
from abc import ABC, abstractmethod

class Space(ABC):

    @abstractmethod
    def sample(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def contains(self, x, *args, **kwargs):
        raise NotImplementedError

    def to_jsonable(self, sample_n):
        return sample_n

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class Box(Space):
    def __init__(self, low, high, shape=None):
        self.low = np.array(low)
        self.high = np.array(high)
        self.shape = shape if shape is not None else self.low.shape

    def sample(self):
        return np.random.uniform(low=self.low, high=self.high, size=self.shape)

    def contains(self, x):
        return np.all(x >= self.low) and np.all(x <= self.high)

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def __repr__(self):
        return f"Box(low={self.low}, high={self.high}, shape={self.shape})"


class Discrete(Space):
    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(self.n)

    def contains(self, x):
        return isinstance(x, int) and 0 <= x < self.n

    def to_jsonable(self, sample_n):
        return int(sample_n)

    def __repr__(self):
        return f"Discrete(n={self.n})"


class MultiDiscrete(Space):
    def __init__(self, nvec):
        self.nvec = np.array(nvec)

    def sample(self):
        return np.random.randint(low=0, high=self.nvec)

    def contains(self, x):
        return np.all(x >= 0) and np.all(x < self.nvec)

    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()

    def __repr__(self):
        return f"MultiDiscrete(nvec={self.nvec})"


class Dict(Space):
    def __init__(self, spaces):
        self.spaces = spaces

    def sample(self):
        return {key: space.sample() for key, space in self.spaces.items()}

    def contains(self, x):
        return all(key in x and self.spaces[key].contains(x[key]) for key in self.spaces)

    def to_jsonable(self, sample_n):
        return {key: space.to_jsonable(sample_n[key]) for key, space in self.spaces.items()}

    def __repr__(self):
        return f"Dict(spaces={self.spaces})"


class Tuple(Space):
    def __init__(self, spaces):
        self.spaces = spaces

    def sample(self):
        return tuple(space.sample() for space in self.spaces)

    def contains(self, x):
        return len(x) == len(self.spaces) and all(space.contains(x[i]) for i, space in enumerate(self.spaces))

    def to_jsonable(self, sample_n):
        return [space.to_jsonable(sample_n[i]) for i, space in enumerate(self.spaces)]

    def __repr__(self):
        return f"Tuple(spaces={self.spaces})"


class Text(Space):
    def __init__(self, max_length):
        self.max_length = max_length

    def sample(self):
        length = np.random.randint(1, self.max_length + 1)
        return ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), length))

    def contains(self, x):
        return isinstance(x, str) and len(x) <= self.max_length

    def to_jsonable(self, sample_n):
        return str(sample_n)

    def __repr__(self):
        return f"Text(max_length={self.max_length})"


class Sequence(Space):
    def __init__(self, space, max_length=None):
        self.space = space
        self.max_length = max_length

    def sample(self):
        length = np.random.randint(1, self.max_length + 1) if self.max_length else np.random.randint(1, 10)
        return [self.space.sample() for _ in range(length)]

    def contains(self, x):
        length_ok = len(x) <= self.max_length if self.max_length else True
        return length_ok and all(self.space.contains(item) for item in x)

    def to_jsonable(self, sample_n):
        return [self.space.to_jsonable(item) for item in sample_n]

    def __repr__(self):
        return f"Sequence(space={self.space}, max_length={self.max_length})"


class Graph(Space):
    def __init__(self, node_space, edge_space):
        self.node_space = node_space
        self.edge_space = edge_space

    def sample(self, max_nodes=10, max_edges=None):
        num_nodes = np.random.randint(1, max_nodes + 1)
        num_edges = np.random.randint(0, max_edges if max_edges is not None else num_nodes * (num_nodes - 1) // 2 + 1)
        nodes = [self.node_space.sample() for _ in range(num_nodes)]
        edges = [self.edge_space.sample() for _ in range(num_edges)]
        return {"nodes": nodes, "edges": edges}

    def contains(self, x):
        nodes_ok = all(self.node_space.contains(node) for node in x["nodes"])
        edges_ok = all(self.edge_space.contains(edge) for edge in x["edges"])
        return nodes_ok and edges_ok

    def to_jsonable(self, sample_n):
        return {
            "nodes": [self.node_space.to_jsonable(node) for node in sample_n["nodes"]],
            "edges": [self.edge_space.to_jsonable(edge) for edge in sample_n["edges"]]
        }

    def __repr__(self):
        return f"Graph(node_space={self.node_space}, edge_space={self.edge_space})"
