from protodata.serialization_ops import SerializeSettings
from protodata.reading_ops import DataSettings
from protodata.utils import create_dir
from protodata.data_ops import NumericColumn, feature_normalize, \
    map_feature_type, float64_feature, int64_feature

import tensorflow as tf
import numpy as np

import os
import logging

QUANTUM_TRAIN_NAME = 'phy_train.dat'
QUANTUM_TEST_NAME = 'phy_test.dat'

MISSING_COLS = [21, 22, 23, 45, 46, 47, 30, 56]
ZERO_COLS = [48, 49, 50, 51, 52]
TO_DELETE_COLS = MISSING_COLS + ZERO_COLS

logger = logging.getLogger(__name__)


class QuantumSerialize(SerializeSettings):

    def __init__(self, data_path):
        """ See base class """
        super(QuantumSerialize, self).__init__(data_path)
        create_dir(data_path)
        if not is_downloaded(data_path):
            raise RuntimeError(
                'This dataset has been extracted from the KDD Cup 2004. ' +
                'In order to process, please proceed to request a copy in ' +
                'http://osmot.cs.cornell.edu/kddcup/datasets.html. After ' +
                'downloading it, place the extracted content into '
                '%s and repeat this operation' % data_path
            )

    def read(self):
        train_data = np.loadtxt(
            get_train_data_path(self.data_path), dtype=float
        )
        test_data = np.loadtxt(
            get_test_data_path(self.data_path), dtype=object
        )

        # The columns we are deleting are those who have missing values
        train_data = np.delete(train_data, TO_DELETE_COLS, axis=1)
        test_data = np.delete(test_data, TO_DELETE_COLS, axis=1)

        train_labels = train_data[:, 1]
        # Add -1 as test label since the information is not public
        test_labels = np.zeros(test_data.shape[0]) - 1

        # Free feature arrays from the label info
        train_data = np.delete(train_data, 1, axis=1)
        test_data = np.delete(test_data, 1, axis=1)

        # Convert test array into float now
        test_data = test_data.astype(float)

        # Append into general objects
        self.features = np.concatenate(
            [train_data, test_data], axis=0
        )
        self.labels = np.concatenate(
            [train_labels, test_labels], axis=0
        )
        self.n_train = train_data.shape[0]

        # Keep id and remove it from features
        self.ids = self.features[:, 0]
        self.features = np.delete(self.features, 0, axis=1)

    def get_validation_indices(self, train_ratio, val_ratio):
        """ Separates data into training, validation and test and normalizes
        the columns by using z-scores """
        logger.warn(
            'Data is already separated between training and test ' +
            'in Quantum Physics dataset. Validation ratio will only be used'
        )

        n_total = self.features.shape[0]
        max_val_ratio = self.n_train / n_total

        if val_ratio >= max_val_ratio:
            raise ValueError(
                'Validation ratio cannot exceed base training ratio. ' +
                'Maximum allowed is %f' % max_val_ratio
            )

        train = range(self.n_train)
        n_val = int(self.n_train * val_ratio)
        val = np.random.permutation(train)[:n_val]
        test = range(self.n_train, self.features.shape[0])

        # Store normalization info
        self.feature_norm = self._normalize_features(train, val)

        return train, val, test

    def _normalize_features(self, train_idx, val_idx):
        training = np.concatenate([train_idx, val_idx])

        mean_c, std_c, min_c, max_c = \
            feature_normalize(self.features[training, :])

        self.features = (self.features - mean_c) / std_c

        # Store normalization info
        return {'mean': mean_c, 'std': std_c, 'min': min_c, 'max_c': max_c}

    def get_options(self):
        options = {'feature_normalization': self.feature_norm}
        return options

    def define_columns(self):
        cols = []

        # Columns
        for i in range(self.features.shape[1]):
            current_col = NumericColumn(
                name=str(i), type=map_feature_type(np.dtype('float'))
            )
            cols.append(current_col)

        # Instance id
        cols.append(NumericColumn(
            name='id', type=map_feature_type(np.dtype('int'))
        ))

        # Label
        cols.append(NumericColumn(
                name='class', type=map_feature_type(np.dtype('int'))
        ))

        return cols

    def build_examples(self, index):
        row = self.features[index, :]
        feature_dict = {}
        for i in range(self.features.shape[1]):
            feature_dict.update(
                {str(i): float64_feature(row[i])}
            )

        feature_dict.update({
            'id': int64_feature(int(self.ids[index]))
        })

        feature_dict.update(
            {'class': int64_feature(int(self.labels[index]))}
        )

        return [tf.train.Example(features=tf.train.Features(feature=feature_dict))]  # noqa


class QuantumSettings(DataSettings):

    def __init__(self, dataset_location, image_specs=None,
                 embedding_dimensions=32, quantizer=None):
        super(QuantumSettings, self).__init__(
            dataset_location=dataset_location,
            image_specs=image_specs,
            embedding_dimensions=embedding_dimensions,
            quantizer=quantizer)

    def tag(self):
        return 'quantum'

    def size_per_instance(self):
        return 0.5

    def target_class(self):
        return 'class'

    def _target_type(self):
        return tf.int32

    def _get_num_classes(self):
        return 2

    def select_wide_cols(self):
        return [v.to_column() for k, v in self.columns.items() if k != 'id']

    def select_deep_cols(self):
        return RuntimeError('No embeddings in this dataset')


def is_downloaded(folder):
    """ Returns whether data has been downloaded """
    return os.path.isfile(get_train_data_path(folder)) \
        and os.path.isfile(get_test_data_path(folder))


def get_train_data_path(folder):
    return os.path.join(folder, QUANTUM_TRAIN_NAME)


def get_test_data_path(folder):
    return os.path.join(folder, QUANTUM_TEST_NAME)
