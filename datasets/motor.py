"""

Dataset for UCI Sensorless Drive Diagnostics

http://archive.ics.uci.edu/ml/datasets/Dataset+for+Sensorless+Drive+Diagnosis#

"""

from protodata.serialization_ops import SerializeSettings
from protodata.reading_ops import DataSettings
from protodata.utils import create_dir
from protodata.data_ops import NumericColumn, feature_normalize, \
    map_feature_type, float64_feature, int64_feature, split_data

import tensorflow as tf
import numpy as np

import os
import logging
from six.moves import urllib

logger = logging.getLogger(__name__)

DATA_FILENAME = 'motor.dat'
DATA_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00325/Sensorless_drive_diagnosis.txt'  # noqa


class MotorSerialize(SerializeSettings):

    def __init__(self, data_path):
        """ See base class """
        super(MotorSerialize, self).__init__(data_path)
        create_dir(data_path)
        if not is_downloaded(data_path):
            logger.info('Downloading Motor dataset ...')
            urllib.request.urlretrieve(DATA_URL, get_data_path(data_path))

    def read(self):
        data = np.loadtxt(get_data_path(self.data_path), dtype=float)
        self.features, self.labels = data[:, :-1], data[:, -1]

    def get_validation_indices(self, train_ratio, val_ratio):
        train, val, test = split_data(
            self.features.shape[0], train_ratio, val_ratio
        )

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

        feature_dict.update(
            {'class': int64_feature(int(self.labels[index]))}
        )

        return [tf.train.Example(features=tf.train.Features(feature=feature_dict))]  # noqa


class MotorSettings(DataSettings):

    def __init__(self, dataset_location, image_specs=None,
                 embedding_dimensions=32, quantizer=None, **params):
        super(MotorSettings, self).__init__(
            dataset_location=dataset_location,
            image_specs=image_specs,
            embedding_dimensions=embedding_dimensions,
            quantizer=quantizer)

    def tag(self):
        return 'motor'

    def size_per_instance(self):
        return 0.5

    def target_class(self):
        return 'class'

    def _target_type(self):
        return tf.int32

    def _get_num_classes(self):
        return 11

    def select_wide_cols(self):
        return [v.to_column() for k, v in self.columns.items()]

    def select_deep_cols(self):
        return RuntimeError('No embeddings in this dataset')


def is_downloaded(folder):
    """ Returns whether data has been downloaded """
    return os.path.isfile(get_data_path(folder)) \
        and os.path.isfile(get_data_path(folder))


def get_data_path(folder):
    return os.path.join(folder, DATA_FILENAME)
