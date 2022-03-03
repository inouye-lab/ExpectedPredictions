from typing import Optional, Tuple, NoReturn

import numpy as np


class DataSet(object):
    _images: np.ndarray
    _labels: np.ndarray
    _one_hot_labels: Optional[np.ndarray]
    _features: Optional[np.ndarray]
    _num_samples: int
    _num_epochs: int
    _index: int

    def __init__(self, images: np.ndarray, labels: np.ndarray, one_hot: bool = True):
        self._images = images
        self._labels = labels
        if one_hot:
            self._one_hot_labels = to_one_hot_encoding(labels)
        else:
            self._one_hot_labels = None
        self._features = None
        self._num_samples = self._images.shape[0]
        self._num_epochs = 0
        self._index = 0

    @property
    def images(self) -> np.ndarray:
        return self._images

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def one_hot_labels(self) -> Optional[np.ndarray]:
        return self._one_hot_labels

    @property
    def features(self) -> Optional[np.ndarray]:
        return self._features

    @features.setter
    def features(self, value: Optional[np.ndarray]) -> NoReturn:
        self._features = value

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @property
    def num_epochs(self) -> int:
        return self._num_epochs

    def next_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return the next `batch_size` examples, features and labels from this data set."""
        assert batch_size <= self._num_samples

        if self._index + batch_size >= self._num_samples:
            perm: np.ndarray = np.arange(self._num_samples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            self._features = self._features[perm]
            self._index = 0
            self._num_epochs += 1

        images = self._images[self._index: self._index + batch_size]
        labels = self._labels[self._index: self._index + batch_size]
        features = self._features[self._index: self._index + batch_size]
        self._index += batch_size
        return images, features, labels, to_one_hot_encoding(labels)


class DataSets(object):
    _train: DataSet
    _test: DataSet
    _valid: DataSet

    def __init__(self, train: DataSet, valid: DataSet, test: DataSet):
        self._train = train
        self._test = test
        self._valid = valid

    @property
    def train(self) -> DataSet:
        return self._train

    @property
    def valid(self) -> DataSet:
        return self._valid

    @property
    def test(self) -> DataSet:
        return self._test


def to_one_hot_encoding(labels: np.ndarray) -> np.ndarray:
    num_classes = np.max(labels) + 1
    one_hot_labels = np.zeros(shape=(len(labels), num_classes), dtype=np.float64)
    for i in range(len(labels)):
        one_hot_labels[i][labels[i]] = 1.0
    return one_hot_labels
