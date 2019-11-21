import numpy as np


class PartitionedDataset:
    """
              0 <= i < infer_start    -> labeled partition
    infer_start <= i < unknown_start  -> infer partition
                   i >= unknown_start -> unknown partition
    """

    def __init__(self, data,data_bis=None, copy=False):
        self.data = data
        self.data_bis = data_bis
        self.__data = data.copy() if copy else data

        self.__size = len(data)

        self.__row_to_index = np.arange(self.__size)
        self._inferred_start = 0
        self._unknown_start = 0

        self.__index_to_row = {i: i for i in range(self.__size)}

        self.__labels = []
        self.__label_tags = []

        self.__previous_inferred_start = 0

    def __len__(self):
        return self.__size

    def __repr__(self):
        return "labeled: {0}, inferred: {1}, unknown: {2}".format(
            self.__row_to_index[:self._inferred_start],
            self.__row_to_index[self._inferred_start:self._unknown_start],
            self.__row_to_index[self._unknown_start:]
        )

    @property
    def indexes(self):
        return self.__row_to_index

    ##################
    # MOVING DATA
    ##################
    def move_to_labeled(self, indexes, labels, tag):
        self.__previous_inferred_start = self._inferred_start

        for idx in indexes:
            self.__move_single(idx, True)

        self.__labels.extend(labels)
        self.__label_tags.extend([tag] * len(labels))

    def move_to_inferred(self, indexes):
        for idx in indexes:
            self.__move_single(idx, False)

    def __move_single(self, idx, to_labeled):
        pos = self.__index_to_row[idx]

        if pos < self._inferred_start:
            raise RuntimeError('Index {0} is already in labeled set.'.format(idx))

        if not to_labeled and self._inferred_start <= pos < self._unknown_start:
            raise RuntimeError('Index {0} is already in inferred set.'.format(idx))

        if pos >= self._unknown_start:
            self.__swap(pos, self._unknown_start)
            
            pos = self._unknown_start
            self._unknown_start += 1

        if to_labeled and pos >= self._inferred_start:
            self.__swap(pos, self._inferred_start)
            self._inferred_start += 1

    def __swap(self, i, j):
        idx_i, idx_j = self.__row_to_index[i], self.__row_to_index[j]
        self.__row_to_index[i], self.__row_to_index[j] = idx_j, idx_i
        self.__index_to_row[idx_i], self.__index_to_row[idx_j] = j, i
        self.__data[[i, j]] = self.__data[[j, i]]
        if self.data_bis is not None:
            self.data_bis[[i,j]] = self.data_bis[[j,i]]

    def remove_inferred(self):
        # flush inferred partition
        self._unknown_start = self._inferred_start

        # copy labeled indexes slice because we will modify list inplace
        labeled_idx = self.__row_to_index[:self._inferred_start].copy()

        for idx, tag in zip(labeled_idx, self.__label_tags):
            if tag != 'user':
                self.__move_right(idx)

        self.__labels = [lb for lb, tag in zip(self.__labels, self.__label_tags) if tag == 'user']
        self.__label_tags = ['user'] * len(self.__labels)

    def __move_right(self, idx):
        pos = self.__index_to_row[idx]

        if pos >= self._unknown_start:
            raise RuntimeError('Index {0} is already in unknown set.'.format(idx))

        if pos < self._inferred_start:
            self.__swap(pos, self._inferred_start - 1)
            pos = self._inferred_start - 1
            self._inferred_start -= 1
            self.__previous_inferred_start -= 1

        if pos < self._unknown_start:
            self.__swap(pos, self._unknown_start - 1)
            self._unknown_start -= 1

    ##################
    # SIZES
    ##################
    @property
    def labeled_size(self):
        return self._inferred_start

    @property
    def infer_size(self):
        return self._unknown_start - self._inferred_start

    @property
    def unknown_size(self):
        return self.__size - self._unknown_start

    @property
    def unlabeled_size(self):
        return self.__size - self._inferred_start

    ##################
    # DATA SLICING
    ##################
    @property
    def labeled(self):
        return self.__row_to_index[:self._inferred_start], self.__data[:self._inferred_start]

    @property
    def inferred(self):
        return self.__row_to_index[self._inferred_start:self._unknown_start], self.__data[self._inferred_start:self._unknown_start]

    @property
    def unknown(self):
        return self.__row_to_index[self._unknown_start:], self.__data[self._unknown_start:]

    @property
    def unlabeled(self):
        return self.__row_to_index[self._inferred_start:], self.__data[self._inferred_start:]

    ##################
    # SAMPLING
    ##################
    def sample_unknown(self, subsample=float('inf')):
        return self.__subsample(subsample, self._unknown_start)

    def sample_unlabeled(self, subsample=float('inf')):
        return self.__subsample(subsample, self._inferred_start)

    def __subsample(self, size, start):
        remaining = self.__size - start

        if remaining == 0:
            raise RuntimeError("There are no points to sample from.")

        if size >= remaining:
            return self.__row_to_index[start:], self.__data[start:]

        row_sample = np.random.choice(np.arange(start, self.__size), size=size, replace=False)
        return self.__row_to_index[row_sample], self.__data[row_sample]

    ##################
    # LABELED DATA
    ##################
    @property
    def last_labeled_set(self):
        return self.__data[self.__previous_inferred_start:self._inferred_start], np.array(self.__labels[self.__previous_inferred_start:self._inferred_start])

    @property
    def training_set(self):
        if self.data_bis is not None:
            step=0.01
            n = 20
#             liste_true = []
#             for x in range(self._inferred_start):
#                 if self.__labels[x]:
#                     liste_true.append(x)
#             true_label = [self.__labels[true] for true in liste_true]
#             true_data = np.array([self.__data[true] for true in liste_true])
#             true_data_bis = np.array([self.data_bis[true] for true in liste_true])
#             print(np.append(np.append(self.__data[:self._inferred_start],                                                              true_data_bis,axis=0),                                                                  0.5*(true_data+true_data_bis),axis=0),                                                        np.append(np.append(np.array(self.__labels),np.array(true_label),       axis=0),np.array(true_label),axis=0))
#             return np.append(self.__data[:self._inferred_start],                                                              true_data_bis,axis=0),                                                        np.append(np.array(self.__labels),np.array(true_label),       axis=0)
#             a =np.append(self.__data[:self._inferred_start],self.data_bis[:self._inferred_start],axis=0)
#             b = np.append(a,(0.95*self.__data[:self._inferred_start]+0.05*self.data_bis[:self._inferred_start]),axis=0)
#             c = np.append(b,(0.9*self.__data[:self._inferred_start]+0.1*self.data_bis[:self._inferred_start]),axis=0)
            y = np.append(np.array(self.__labels),np.array(self.__labels))
            x = np.append(self.__data[:self._inferred_start],self.data_bis[:self._inferred_start],axis=0)
            for i in range(n):
                x = np.append(x, (1-i*step)*self.__data[:self._inferred_start]+i*step*self.data_bis[:self._inferred_start],axis=0)
            for i in range(n):            
                y = np.append(y, np.array(self.__labels),axis=0)
            
            return x,y
        else:
            return self.__data[:self._inferred_start], np.array(self.__labels)
