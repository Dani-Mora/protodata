from protodata.serialization_ops import SerializeSettings
from protodata.reading_ops import DataSettings
from protodata.utils import create_dir
from protodata.data_ops import NumericColumn, split_data, feature_normalize, \
    map_feature_type, float64_feature, int64_feature

import tensorflow as tf
import numpy as np

import os
import logging

QUANTUM_TRAIN_NAME = 'phy_train.dat'
QUANTUM_TEST_NAME = 'phy_test.dat'

TO_DELETE_COLS = [21, 22, 23, 45, 46, 47, 30, 56]

logger = logging.getLogger(__name__)

# TODO: it is important to remember that the first column is the label


class QuantumSerialize(SerializeSettings):

    def __init__(self, data_path):
        """ See base class """
        super(QuantumSerialize, self).__init__(data_path)
        create_dir(data_path)
        if not is_downloaded(data_path):
            raise RuntimeError(
                'This dataset has been extracted from the KDD Cup 2004' +
                'In order to process, please proceed to request a copy in ' +
                'http://osmot.cs.cornell.edu/kddcup/datasets.html. After ' +
                'downloading it, place the extracted content into '
                '%s and repeat this operation'
            )

    def read(self):
        self.train_data = np.loadtxt(
            get_train_data_path(self.data_path), dtype=float
        )
        self.test_data = np.loadtxt(
            get_train_data_path(self.data_path), dtype=float
        )

        # The columns we are deleting are those who have missing values
        self.train_data = np.delete(self.train_data, TO_DELETE_COLS, axis=1)
        self.test_data = np.delete(self.test_data, TO_DELETE_COLS, axis=1)

        self.train_labels = self.train_data[:, 1]
        self.test_labels = self.test_data[:, 1]

        # Free feature arrays from the label info
        self.train_data = np.delete(self.train_data, 1, axis=1)
        self.test_data = np.delete(self.test_data, 1, axis=1)

    def get_validation_indices(self, train_ratio, val_ratio):
        """ Separates data into training, validation and test and normalizes
        the columns by using z-scores """
        logger.warn(
            'Data is already separated between training and test ' +
            'in Quantum Physics dataset. Validation ratio will only be used'
        )

        n_total = self.train_data.shape[0] + self.test_data.shape[0]
        max_val_ratio = self.train_data.shape[0] / n_total

        if val_ratio >= max_val_ratio:
            raise ValueError(
                'Validation ratio cannot exceed base training ratio. ' +
                'Maximum allowed is %f' % max_val_ratio
            )

        # Fuse both datasets and provide indexes
        train, val, test = split_data(self.features.shape[0],
                                      train_ratio,
                                      val_ratio)

        # Store normalization info
        self.feature_norm = self._normalize_features(train, val)

        return train, val, test

    def _normalize_features(self, train_idx, val_idx):
        training = np.concatenate([train_idx, val_idx])
        mean_c, std_c, min_c, max_c = \
            feature_normalize(self.features.iloc[training, :])

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

        # Label
        cols.append(NumericColumn(
                name='class', type=map_feature_type(np.dtype('int'))
        ))

        return cols

    def build_examples(self, index):
        row = self.features.iloc[index, :]
        feature_dict = {}
        for i in range(self.features.shape[1]):
            feature_dict.update(
                {str(i): float64_feature(row.iloc[i])}
            )

        feature_dict.update(
            {'class': int64_feature(int(self.labels.iloc[index]))}
        )

        return [tf.train.Example(features=tf.train.Features(feature=feature_dict))]  # noqa


class AusSettings(DataSettings):

    def __init__(self, dataset_location, image_specs=None,
                 embedding_dimensions=32, quantizer=None):
        super(AusSettings, self).__init__(
            dataset_location=dataset_location,
            image_specs=image_specs,
            embedding_dimensions=embedding_dimensions,
            quantizer=quantizer)

    def tag(self):
        return 'aus'

    def size_per_instance(self):
        return 0.5

    def target_class(self):
        return 'class'

    def _target_type(self):
        return tf.int32

    def _get_num_classes(self):
        return 2

    def select_wide_cols(self):
        return [v.to_column() for k, v in self.columns.items()]

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
