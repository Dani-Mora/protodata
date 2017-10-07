"""
Class for scikitlearn integrated UCI datasets used for testing
"""

from abc import ABCMeta
import abc

from sklearn.datasets import load_diabetes
from sklearn.datasets import load_boston

from protodata.data_ops import NumericColumn, split_data, feature_normalize, \
    map_feature_type, map_feature
from protodata.serialization_ops import SerializeSettings
from protodata.reading_ops import DataSettings
from protodata.utils import get_logger

import numpy as np
import tensorflow as tf

logger = get_logger(__name__)


class ScikitSerialize(SerializeSettings):

    __metaclass__ = ABCMeta

    def __init__(self):
        super(ScikitSerialize, self).__init__(None)
        self.features, self.labels = None, None
        self.feature_norm = None

    def read(self):
        dataset = self.get_dataset()
        self.features = np.array(dataset.data)
        self.labels = np.array(dataset.target)

    @abc.abstractmethod
    def get_dataset(self):
        """ Returns the specific dataset data as a tuple
         containing 'data' and 'target'
         """

    @abc.abstractmethod
    def get_column_names(self):
        """ Returns the names of the columns of the dataset """

    @abc.abstractmethod
    def get_target_name(self):
        """ Returns the name of the target column"""

    def get_validation_indices(self, train_ratio, val_ratio):
        train, val, test = split_data(self.features.shape[0],
                                      train_ratio,
                                      val_ratio)

        # Normalize given training data (train + validation)
        training = np.concatenate([train, val])
        mean_c, std_c, min_c, max_c = \
            feature_normalize(self.features[training, :])
        self.features = (self.features - mean_c) / std_c

        # Store normalization info
        self.feature_norm = {'mean': mean_c,
                             'std': std_c,
                             'min': min_c,
                             'max_c': max_c}
        return train, val, test

    def get_options(self):
        return {'norm': self.feature_norm}

    def build_examples(self, index):
        """ See base class"""
        feature_dict = {}

        # Features
        for i, name in enumerate(self.get_column_names()):
            mapped = map_feature(self.features[index, i],
                                 self.features[:, i].dtype)
            feature_dict.update({name: mapped})

        # Index
        feature_dict.update({'index': map_feature(index, type(index))})

        # Label
        mapped_label = map_feature(self.labels[index], self.labels.dtype)
        feature_dict.update({self.get_target_name(): mapped_label})

        return [tf.train.Example(features=tf.train.Features(feature=feature_dict))] # NOQA

    def define_columns(self):
        """ See base class """
        columns = []
        for i in range(self.features.shape[1]):
            numeric_type = map_feature_type(self.features[:, i].dtype)
            columns.append(NumericColumn(self.get_column_names()[i],
                                         type=numeric_type))

        # Add column for index
        columns.append(NumericColumn('index',
                                     type=map_feature_type(np.int)))

        # Add column for label
        columns.append(NumericColumn(self.get_target_name(),
                                     type=map_feature_type(np.float)))

        return columns


class ScikitSettings(DataSettings):

    __metaclass__ = ABCMeta

    def __init__(self, dataset_location, image_specs=None,
                 embedding_dimensions=32, quantizer=None):
        super(ScikitSettings, self).__init__(
            dataset_location=dataset_location,
            image_specs=image_specs,
            embedding_dimensions=embedding_dimensions,
            quantizer=quantizer)

    def size_per_instance(self):
        return 1

    def _target_type(self):
        return tf.float32

    def select_wide_cols(self):
        return [v.to_column()
                for k, v in self.columns.items() if k != 'index']

    def select_deep_cols(self):
        return []


"""
    Boston dataset
"""


class BostonSerialize(ScikitSerialize):

    __metaclass__ = ABCMeta

    def __init__(self):
        super(BostonSerialize, self).__init__()

    def get_dataset(self):
        return load_boston()

    def get_column_names(self):
        return ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS',
                'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

    def get_target_name(self):
        return 'MEDV'


class BostonSettings(ScikitSettings):

    def target_class(self):
        return 'MEDV'

    def _get_num_classes(self):
        return 1


"""
    Diabetes dataset
"""


class DiabetesSerialize(ScikitSerialize):

    __metaclass__ = ABCMeta

    def __init__(self):
        super(DiabetesSerialize, self).__init__()

    def get_dataset(self):
        return load_diabetes()

    def get_column_names(self):
        return ['feature' + str(i) for i in range(10)]

    def get_target_name(self):
        return 'target'


class DiabetesSettings(ScikitSettings):

    def target_class(self):
        return 'target'

    def _get_num_classes(self):
        return 1
