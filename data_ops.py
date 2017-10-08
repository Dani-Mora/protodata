import matplotlib.path as mplPath
from abc import ABCMeta
import abc

from protodata.utils import read_json, get_logger
from protodata.columns import create_image_column

import numpy as np
import os
import tensorflow as tf

logger = get_logger(__name__)


""" General functions for data manipulation """


class TrainMode(object):

    WIDE = 'wide'
    DEEP = 'deep'
    CNN = 'cnn'
    WIDE_AND_DEEP = 'wide_and_deep'
    ALL = 'wide_deep_cnn'


class DataMode(object):

    TRAINING = 'training'
    VALIDATION = 'validation'
    TEST = 'testing'


""" Data filename pattern """


def get_filename(name_tag, shard, num_shards):
    """ Returns the format of the record file names given the
    data tag, the current shard and the total amount of shards"""
    return '%s-%.5d-of-%.5d' % (name_tag, shard, num_shards)


def get_filename_pattern(folder, tag):
    """ Returns the pattern to read record files """
    return os.path.join(folder, '%s-*' % str(tag))


""" Neighborhood processing functions """


def read_city_data(path):
    """ Maps the neighborhoods of the city from the provided data
    so each entry contains a map between the neighborhood and the
    polygon delimiting it """
    raw_cities = read_json(path)
    cities = {}
    for n in raw_cities['features']:
        coords = np.array([c for c in n['geometry']['coordinates'][0][0]])

        # Make sure coordinates have form Nx2
        if coords.shape[1] == 3:
            coords = coords[:, :-1]

        if coords.shape[1] != 2:
            raise ValueError('Coordinates have depth %d and should have 2'
                             % coords.shape[1])

        # Map into polygon and add to dictionary
        neigh_name = n['properties']['neighbourhood']
        bb = mplPath.Path(coords)
        cities.update({neigh_name: bb})

    return cities


def get_neighborhood(neighs, longitude, latitude):
    """ Returns the neighborhood of the input point. We assume that a
    point can only be contained in one neighborhood (otherwise data is broken)
    Args:
        neighs: Dictionary mapping neighborhoods and their area
        longitude: Longitude of the input point
        latitude: Latitude of the input point
    Returns:
        name: Name of the neighborhood the point belongs to or None if no
            valid neighborhood found
    """
    for neigh, bb in neighs.items():
        if bb.contains_point((longitude, latitude)):
            return neigh
    return None


""" Data normalization functions """


def quantile_norm(val, edges, nq):
    """ Normalizes continuous values so they are converted by the formula:

            x = i/(nq - 1)

            Where i is the quartile number [0, nq) the input value falls in
            the feature distribution and nq is the number of quartiles used.
    Args:
        val: Input value
        edges: Histogram edges
        nq: Number of quantiles
    """
    if val >= edges[-1]:
        quantile = len(edges)
    elif val <= edges[0]:
        quantile = 0
    else:
        quantile = np.argmax(edges >= val)

    return quantile / (nq - 1.0)


def feature_normalize(dataset):
    """ Normalizes feature and returns the mean, deviation and minimum and
    maximum values """
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    min_c, max_c = np.min(dataset, axis=0), np.max(dataset, axis=0)
    return mu, sigma, min_c, max_c


""" Tensorflow helpers """


def get_interval_mask(low, high, values):
    """ Returns the boolean mask such that True values are those x which
     are in interval low <= x < high """
    # Since greater_equal and less support broadcasting, we can do this
    low_t, high_t = tf.greater_equal(values, low), tf.less(values, high)
    return tf.logical_and(low_t, high_t)


def copy_columns(x, num):
    """ Replicates the input tensor x by concatenating its columns num times"""
    # Create replicas into a 1D vector
    vector = tf.tile(x, tf.stack([num]))
    # Transpose and reshape
    return tf.transpose(tf.reshape(vector, tf.stack([num, -1])))


""" Pandas helpers """


def get_column_info(data, excluded=[]):
    """ Returns the zipped pair of columns and corresponding numpy type
    for columns of interest  """
    col_names = [i for i in data.columns.values if i not in excluded]
    col_types = [data[i].dtype for i in col_names if i not in excluded]
    return list(zip(col_names, col_types))


def quantile_normalization(data, train_ind, nq, excluded=[]):
    """ Normalizes into [0,1] numeric values in the dataset using
    quantile normalization as described in:

        Wide & Deep Learning for Recommender Systems.
        Cheng et al (2016)
        [https://arxiv.org/abs/1606.07792]

    Args:
        data: Pandas dataframe
        train_ind: Instances in the training set
        nq: Number of quantiles to use
        excluded: Columns to ignore in the normalization process
    Returns:
        data: Normalized dataset given the mean and standard deviation
            extracted from the training
        d: Dictionary indexed by column name where each entry contains
            the cuts for a numeric feature
    """
    d = {}
    for (name, dtype) in get_column_info(data, excluded=excluded):

        if is_numeric(dtype):

            logger.debug('Normalizing column %s' % name)

            # Compute histogram edges for training data
            train_content = data.iloc[train_ind, name]
            hist, edges = np.histogram(train_content, bins=nq-2)
            edges = np.array(edges)

            # Store entry in dictionary
            d[name] = edges

            # Update column
            data[name] = data[name].apply(lambda x: quantile_norm(x,
                                                                  edges,
                                                                  nq))

    return data, d


def z_scores(data, train_ind, excluded=[]):
    """ Returns a dictionary containing the min, max, mean and standard
        deviation for the numeric
    columns in the dataframe
    Args:
        data: Pandas dataframe
        train_ind: List of instance indices belonging to the train set
        excluded: Columns to ignore in the normalization process
    Returns:
        data: Normalized dataset given the mean and standard deviation
            extracted from the training
        d: Dictionary indexed by numeric column name where each entry
            contains its mean and std
    """
    d = {}
    for (name, dtype) in get_column_info(data, excluded=excluded):

        if is_numeric(dtype):

            # Compute mean and std on training
            train_content = data.iloc[train_ind, name].as_matrix()
            mean, std, min_c, max_c = feature_normalize(train_content)

            # Store entry in dictionary
            d[name] = {'mean': mean, 'std': std, 'min': min_c, 'max': max_c}

            # Update column
            data[name] = (data[name] - mean)/std

    return data, d


def normalize_data(data, train_ind, zscores=True, excluded=[], nq=5):
    """ Normalizes the input
    Args:
        data: Pandas dataframe
        train_ind: List of instance indices belonging to the train set
        zscores: Whether to use z-scores normalization (True) or
            quantile normalization (False)
        excluded: Columns to ignore in the normalization process
        nq: Number of quantiles. Only used if zscores is False.
    Returns:
        data: Normalized dataset
        d: Dictionary containing metadata of the normalization. For z-scores
            contains mean, min, max and standard deviation of each numeric
            column. For quantile norm, it contains the edges of the histogram
    """
    if zscores:
        return z_scores(data, train_ind=train_ind, excluded=excluded)
    else:
        return quantile_normalization(data,
                                      train_ind=train_ind,
                                      nq=nq,
                                      excluded=excluded)


def to_dummy(data, name):
    """ Converts categorical column into dummy binary column
    (one for each possible value) """

    def is_equal(x, other):
        return x == other

    logger.debug('Converting %s into dummy column ...' % name)

    for val in data[name].unique():

        # Obtain a valid string formatted name for the new column
        col_name = '_'.join([name, val])
        dummy_name = erase_special(unicode_to_str(col_name))

        # Convert according to whether it is equal or not to current value
        data[dummy_name] = data[name].apply(lambda x: is_equal(x, val))

    # Erase original column
    return data.drop(name, axis=1)


def convert_to_dummy(data, excluded_columns=[]):
    """ Converts categorical variables into dummy binary variables
    Args:
        data: pandas dataframe
        excluded_columns: columns to ignore.
    """
    for (feat_name, feat_type) in get_column_info(data,
                                                  excluded=excluded_columns):
        if feat_type == np.dtype('object'):
            data = to_dummy(data, feat_name)
    return data


def convert_boolean(data, excluded_columns=[], func=float):
    """ Converts the boolean columns in the dataset into the desired type
    Args:
        data: pandas Dataframe
        excluded_columns: Columns to exclude in the conversion process
        func: Type (and function) to use to convert booleans to
            (e.g. float, int). Default is float
    """
    for (c, t) in get_column_info(data, excluded=excluded_columns):
        if t == np.dtype('bool'):
            logger.debug('Converting boolean column %s to numeric' % c)
            data[c] = data[c].apply(func)
    return data


def is_categorical(type_def):
    """ Whether input column type represents a categorical
    column (True) or not (False) """
    return type_def == np.dtype('object')


def is_numeric(type_def):
    """ Whether columns represents a numerical value (True) or not (False) """
    return np.issubdtype(type_def, np.number)


def is_bool(type_def):
    """ Whether columns represents a boolean value """
    return np.issubdtype(type_def, bool) and not np.issubdtype(type_def,
                                                               np.number)


""" TF Serialization wrapper operations """


def int64_feature(value):
    """ Wrapper for int64 proto features """
    value = [value] if not isinstance(value, list) else value
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float64_feature(value):
    """ Wrapper for floay64 proto features """
    value = [value] if not isinstance(value, list) else value
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    """ Wrapper for byte proto features """
    # Ensure we have a list of elements
    value = [value] if not isinstance(value, list) else value
    # Convert string into bytes if found
    for i in range(len(value)):
        if isinstance(value[i], str):
            value[i] = str.encode(value[i])
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def map_feature(value, f_type):
    """ Builds the Tensorflow feature for the given feature information """
    if f_type == np.dtype('object'):
        return bytes_feature(value)
    elif f_type == np.dtype('int'):
        return int64_feature(value)
    elif f_type == np.dtype('float'):
        return float64_feature(value)
    elif f_type == np.dtype('bool'):
        return int64_feature(value.astype('int'))
    else:
        raise ValueError('Do not know how to store value {} with type {}'
                         .format(value, f_type))


def map_feature_type(np_type):
    """ Maps numpy types into accepted Tensorflow feature
    types (int64, float32 and string) """
    if np.issubdtype(np_type, np.integer):
        return tf.int64
    elif np.issubdtype(np_type, np.float):
        return tf.float32
    elif np_type == np.bool:
        return tf.int64
    elif np_type == np.object:
        return tf.string
    else:
        raise TypeError('Could not map type {} into a '
                        + 'correct Tensorflow type'.format(np_type))


def str_to_unicode(x):
    """ Encodes input values into UTF-8 """
    if isinstance(x, list):
        return [to_utf8(i) for i in x]
    else:
        return to_utf8(x)


def to_utf8(x):
    """ Encodes the input value into UTF-8 """
    if isinstance(x, unicode):  # noqa
        return x  # Already unicode, no need for conversion
    elif isinstance(x, str):
        return x.decode('utf-8')
    else:
        raise ValueError('Input value must be in string format')


def erase_special(x):
    """ Removes special characters and spaces from the input string """
    return ''.join([i for i in x if i.isalnum()])


def unicode_to_str(x):
    """ Encodes input values into UTF-8 """
    if isinstance(x, list):
        return [to_str(i) for i in x]
    else:
        return to_str(x)


def to_str(x):
    """ Encodes the input value into UTF-8 """
    if isinstance(x, str):
        return x  # Already string, no need for conversion
    elif isinstance(x, unicode):  # noqa
        return x.encode('utf-8')
    else:
        raise ValueError('Input value must be in unicode format')


def split_data(total, train_ratio, val_ratio):
    """ Splits the total number of instances into training, validation
     and testing and returns the corresponding indices for each set """
    # Compute instance number per set
    train_num = int(total * train_ratio)
    val_num = int(total * val_ratio)
    # Get random ordering and extract indices
    permutation = np.random.permutation(total)
    train = permutation[:train_num]
    val = permutation[train_num:train_num+val_num]
    test = permutation[train_num+val_num:]
    return train, val, test


class ExampleColumn(object):

    """ Represents a column from a serializes proto Example """

    __metaclass__ = ABCMeta

    def __init__(self, name, type):
        self.name = name
        self.type = type

    @abc.abstractmethod
    def get_feature(self):
        """ Returns the feature type according to each column """

    @abc.abstractmethod
    def to_column(self, args=None):
        """ Returns the categorical definition of the column """


class NumericColumn(ExampleColumn):

    """ Real-valued column """

    def __init__(self, name, type, length=1):
        """ Args:
            length: Length of the feature value. Default is 1
        """
        super(NumericColumn, self).__init__(name, type)
        self.type = type
        self.length = length

    def get_feature(self):
        return tf.FixedLenFeature([self.length], self.type)

    def to_column(self, args=None):
        return tf.contrib.layers.real_valued_column(self.name,
                                                    self.length)

    def categorize(self, args=None):
        real_valued = tf.contrib.layers.real_valued_column(self.name)
        return tf.contrib.layers.bucketized_column(real_valued,
                                                   boundaries=args)


class SparseColumn(ExampleColumn):

    """ Categorical column """

    def __init__(self, name, type, keys=5):
        """
        Args:
            keys: Number of hash buckets to use or list of keys
                present in the column.
        """
        super(SparseColumn, self).__init__(name, type)
        self.keys = keys

    def get_feature(self):
        return tf.VarLenFeature(self.type)

    def to_column(self, args=None):
        if isinstance(self.keys, list):
            return tf.contrib.layers.sparse_column_with_keys(self.name,
                                                             self.keys)
        elif isinstance(self.keys, int):
            return tf.contrib.layers.sparse_column_with_hash_bucket(self.name,
                                                                    self.keys)
        else:
            raise ValueError('Unexpected type of keys/bucket size of {}'
                             .format(self.keys))

    def to_embedding(self, dims):
        """ Returns an embedding column out of the Sparse column """
        return tf.contrib.layers.embedding_column(self.to_column(),
                                                  dimension=dims)


class ImageColumn(ExampleColumn):

    """ Column that represents an image input """

    def __init__(self, name, format):
        super(ImageColumn, self).__init__(name, tf.string)
        self.format = format

    def get_feature(self):
        return tf.FixedLenFeature([], dtype=self.type)

    def to_column(self, args=None):
        return create_image_column(name=self.name, dims=args)


def create_cross_column(cols, bucket_size=int(1e4)):
    """ Returns a cross column that has the values of the cartesian product of the
    input categorical columns """
    return tf.contrib.layers.crossed_column(cols, hash_bucket_size=bucket_size)
