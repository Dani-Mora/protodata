"""

Fashion MNIST. More information:

https://github.com/zalandoresearch/fashion-mnist

"""

from protodata.serialization_ops import SerializeSettings
from protodata.reading_ops import DataSettings
from protodata.data_ops import bytes_feature, int64_feature, DataMode, \
    map_feature_type, NumericColumn, SparseColumn, ImageColumn
from protodata.utils import create_dir, get_filename_url

import numpy as np
import os
import tempfile
from six.moves import urllib
import gzip
import struct
import array
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)

TRAIN_DATA_URL = \
    'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz'  # noqa
TRAIN_LABELS_URL = \
    'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz'  # noqa
TEST_DATA_URL = \
    'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz'  # noqa
TEST_LABELS_URL = \
    'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'  # noqa

DATA_PARAMS = [16, "B", ">IIII"]
LABELS_PARAMS = [8, "b", ">II"]


class FashionMnistSerialize(SerializeSettings):

    def __init__(self, data_path):
        """ See base class """
        super(FashionMnistSerialize, self).__init__(data_path)
        create_dir(data_path)
        # On-demand download if it does not exist
        if not is_downloaded(data_path):
            logger.info('Downloading Fashion MNIST dataset ...')
            download(TRAIN_DATA_URL, TRAIN_LABELS_URL, data_path)
            download(TEST_DATA_URL, TEST_LABELS_URL, data_path)

    def read(self):

        # Read downloaded data
        train_data = get_dataset(self.data_path, TRAIN_DATA_URL)
        train_labels = get_dataset(self.data_path, TRAIN_LABELS_URL)
        test_data = get_dataset(self.data_path, TEST_DATA_URL)
        test_labels = get_dataset(self.data_path, TEST_LABELS_URL)

        # Load numpy arrays
        train_img, train_labels, n = self.read_numpy(train_data, train_labels)
        test_img, test_labels, ntest = self.read_numpy(test_data, test_labels)

        # Set number of instances, image dimensions and data itself
        self.height, self.width = train_img.shape[1], train_img.shape[2]
        self.ntrain, self.ntest = n, ntest
        self.train_data = train_img
        self.train_labels = train_labels
        self.test_data = test_img
        self.test_labels = test_labels

    @staticmethod
    def _read_raw_array(path, size, type, unpack_format):
        """ Reads an raw array according to its format """
        with open(path, 'rb') as f:
            metadata = struct.unpack(unpack_format, f.read(size))
            ret = array.array(type, f.read())
        return ret, metadata

    @staticmethod
    def _read_raw_data(p):
        """
        :param p: Path to the raw data file
        :return: Numpy format of the data in the path
        """
        return FashionMnistSerialize._read_raw_array(p, *DATA_PARAMS)

    @staticmethod
    def _read_raw_labels(p):
        """
        :param p: Path to the raw label file
        :return: Numpy format of the labels in the path
        """
        return FashionMnistSerialize._read_raw_array(p, *LABELS_PARAMS)

    @staticmethod
    def read_numpy(datap, labelp):
        """ Reads raw data into numpy arrays
        :param datap: Data file path
        :param labelp: Label file path
        :return: Numpy formatted dataset: images, labels
        """

        data, metadata = FashionMnistSerialize._read_raw_data(datap)
        labels, _ = FashionMnistSerialize._read_raw_labels(labelp)
        _, _, rows, cols = metadata

        # Convert into numpy
        n = len(labels)
        imgs = np.zeros((n, rows, cols), dtype=np.uint8)
        lbls = np.zeros((n, 1), dtype=np.int8)
        for i in range(n):
            imgs[i] = np.array(data[i * rows * cols:(i + 1) * rows * cols]) \
                .reshape((rows, cols))
            lbls[i] = labels[i]
        return imgs, lbls, n

    def get_validation_indices(self, train_ratio, val_ratio):
        """ Mnist has already the partition between training and test.
        Only validation ratio is used here """
        logger.warn('MNIST dataset is already split')

        if val_ratio > 0.85:
            raise ValueError('MNIST only support validation up to 85%')

        # Get subset for validation set
        num_validation = int((self.ntrain + self.ntest) * val_ratio)
        num_train = self.ntrain - num_validation

        perm_train = np.random.permutation(self.ntrain)

        train = list(zip(perm_train[:num_train].tolist(),
                     [DataMode.TRAINING] * num_train))

        val = list(zip(perm_train[num_validation:].tolist(),
                   [DataMode.VALIDATION] * num_validation))

        # Test set already computed
        test = list(zip(range(self.ntest), [DataMode.TEST] * self.ntest))

        return train, val, test

    def get_options(self):
        options = {
            'src_height': self.height,
            'src_width': self.width}
        return options

    def define_columns(self):
        return [
            NumericColumn('label',
                          type=map_feature_type(np.dtype('int'))),
            ImageColumn('image', format='JPEG'),
            SparseColumn('colorspace',
                         map_feature_type(np.dtype('object')))
        ]

    def build_examples(self, index):

        # Get row
        i, subset = index
        i = int(i)
        img = self.train_data[i, ...] \
            if subset in [DataMode.TRAINING, DataMode.VALIDATION] \
            else self.test_data[i, ...]
        label = self.train_labels[i, ...] \
            if subset in [DataMode.TRAINING, DataMode.VALIDATION]  \
            else self.test_labels[i, ...]

        # Fill features
        feature_dict = {
            'image': bytes_feature(self.process_image_bytes(img)),
            'label': int64_feature(int(label)),
            'colorspace': bytes_feature('RGB')
        }

        return [tf.train.Example(features=tf.train.Features(feature=feature_dict))]  # noqa


class FashionMnistSettings(DataSettings):

    def __init__(self, dataset_location, image_specs=None,
                 embedding_dimensions=32, quantizer=None):
        super(FashionMnistSettings, self).__init__(
            dataset_location=dataset_location,
            image_specs=image_specs,
            embedding_dimensions=embedding_dimensions,
            quantizer=quantizer)

    def tag(self):
        return 'fashion_mnist'

    def size_per_instance(self):
        return 0.5

    def target_class(self):
        return 'label'

    def _target_type(self):
        return tf.int32

    def _get_num_classes(self):
        return 10

    def image_field(self):
        return 'image'

    def select_wide_cols(self):
        return []

    def select_deep_cols(self):
        return [v.to_column() for (k, v) in self.columns.items()
                if isinstance(v, NumericColumn)]


def is_downloaded(folder):
    """ Returns whether Fashion MNIST has been downloaded """
    paths = [TRAIN_DATA_URL, TRAIN_LABELS_URL, TEST_DATA_URL, TEST_LABELS_URL]
    for i in paths:
        out_path = os.path.join(folder,
                                os.path.splitext(os.path.basename(i))[0])
        if not os.path.isfile(out_path):
            return False
    return True


def get_dataset(folder, url):
    """ Returns the corresponding file name for the desired dataset """
    return os.path.join(folder, get_filename_url(url)[0])


def download(data, labels, outp):
    """ Downloads and extracts the given data in the desired location
    Args:
        data: Url of the data to download for the set
        labels: Url of the labels to download for the set
        outp: Output path where to store the data
    """

    def download_and_extract(url, folder):
        """ Downloads the given url and extracts it in
        the desired location """

        # Download into temp file
        fd, down = tempfile.mkstemp()
        urllib.request.urlretrieve(url, down)

        # Move bytes from input to output
        dst = get_dataset(folder, url)
        with gzip.open(down, 'rb') as infile:
            with open(dst, 'wb') as outfile:
                for line in infile:
                    outfile.write(line)
        # Delete tmp file
        os.close(fd)
        os.remove(down)

    download_and_extract(data, outp)
    download_and_extract(labels, outp)
