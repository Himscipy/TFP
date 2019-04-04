import numpy as np


class minibatcher:
    def __init__(self, data, batch_size, shuffle=True):
        self.x_ds, self.y_ds = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._reset()

    def __len__(self):
        return len(self.idx_pool)

    def _reset(self):
        self.pointer = 0
        idx_list = np.arange(len(self.x_ds))
        if self.shuffle:
            idx_list = np.random.permutation(idx_list)
        self.idx_pool = [idx_list[i: i + self.batch_size]
                         for i in range(0, len(self.x_ds), self.batch_size)]

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.idx_pool):
            self._reset()
            # raise StopIteration()

        idx = self.idx_pool[self.pointer]
        x = self.x_ds[idx]
        y = self.y_ds[idx]
        self.pointer += 1
        return (x, y)
