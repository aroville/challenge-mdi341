import numpy as np
from sklearn import preprocessing

DATA_DIR = '/home/axel/challenge-mdi341/data/'
IMG_SIZE = 48 * 48
LBL_SIZE = 128


class DataSet:
    def __init__(self, suffix, n_img, include_test=True, scaler=None):
        self.suffix = suffix
        self.n_img = n_img
        self.scaler = scaler
        self.images = self.load_data_from_file('data', IMG_SIZE, np.uint8)
        if include_test:
            self.labels = self.load_data_from_file('fv', LBL_SIZE, np.float32)
        self._index_in_epoch = 0
        self.epochs_completed = 0

    def load_data_from_file(self, prefix, size, dtype):
        """
        Load data from the file located in DATA_DIR and specified by prefix
        end suffix. The data is reshaped to a numpy array of size 
        "number of images or labels" * "size of one image or label"
        
        :param prefix: 'data' or 'fv', depending on data being train or test
        :param size: Size of a single data element
        :param dtype: Type of data expected
        :return: A numpy array of dimension (self.n_img * size)
        """
        n_img = self.n_img
        file = DATA_DIR + '{}_{}.bin'.format(prefix, self.suffix)
        with open(file, 'rb') as f:
            args = {'file': f, 'dtype': dtype, 'count': n_img * IMG_SIZE}
            d = np.fromfile(**args).astype(np.float32).reshape(n_img, size)
            if prefix == 'data':
                if self.scaler is None:
                    self.scaler = preprocessing.StandardScaler().fit(d)
                return self.scaler.transform(d)
            return d

    def next_batch(self, batch_size=100):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.n_img:
            # Finished epoch
            self.epochs_completed += 1
            # print('Finished epoch number', self._epochs_completed)

            # Shuffle the data
            perm = np.arange(self.n_img)
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.labels = self.labels[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self.n_img
        end = self._index_in_epoch
        return self.images[start:end], self.labels[start:end]