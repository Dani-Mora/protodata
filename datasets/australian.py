from protodata.serialization_ops import SerializeSettings
from protodata.reading_ops import DataSettings
from protodata.utils import create_dir
from protodata.data_ops import NumericColumn, split_data, feature_normalize, \
    map_feature_type, float64_feature, int64_feature

import tensorflow as tf
import pandas as pd
import numpy as np

import tempfile
import os
import logging
from six.moves import urllib


DATA_FILE_NAME = 'australian.npy'
DATA_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat'  # noqa
CATEGORICAL_IDX = [3, 4, 5, 11]  # 0, 7, 8 and 10 are already dummy variables
logger = logging.getLogger(__name__)


class AusSerialize(SerializeSettings):

    def __init__(self, data_path):
        """ See base class """
        super(AusSerialize, self).__init__(data_path)
        create_dir(data_path)
        # On-demand download if it does not exist
        if not is_downloaded(data_path):
            logger.info('Downloading Australian dataset ...')
            download(DATA_URL, get_data_path(data_path))

    def read(self):
        self.data = np.loadtxt(get_data_path(self.data_path), dtype=float)

        self.features = pd.DataFrame(self.data[:, :-1])

        # Some columns are numerical versions of categorical columns
        # Let's create dummy cars instead
        for idx in CATEGORICAL_IDX:
            dummies = pd.get_dummies(self.features.iloc[:, idx])
            self.features = pd.concat([self.features, dummies], axis=1)

        # Discard original ones
        preserve = [i for i in range(self.features.shape[1])
                    if i not in CATEGORICAL_IDX]
        self.features = self.features.iloc[:, preserve].as_matrix()

        self.labels = self.data[:, -1]

    def get_instance_num(self):
        return self.features.shape[0]

    def get_validation_indices(self, train_ratio, val_ratio):
        """ Separates data into training, validation and test and normalizes
        the columns by using z-scores """

        train, val, test = split_data(self.features.shape[0],
                                      train_ratio,
                                      val_ratio)

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
    return os.path.isfile(get_data_path(folder))


def get_data_path(folder):
    return os.path.join(folder, DATA_FILE_NAME)


def download(url, dst):
    """ Downloads the data file into the given path """
    # Download into temp file
    fd, down = tempfile.mkstemp()
    urllib.request.urlretrieve(url, down)

    # Move bytes from input to output
    data_array = np.loadtxt(down, dtype=float)
    np.savetxt(dst, data_array)

    # Delete tmp file
    os.close(fd)
    os.remove(down)
