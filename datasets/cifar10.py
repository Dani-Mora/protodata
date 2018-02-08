"""

Classic image classification dataset containing 60k images
from 10 different objects

"""

from protodata.serialization_ops import SerializeSettings
from protodata.reading_ops import DataSettings
from protodata.data_ops import bytes_feature, int64_feature, NumericColumn, \
                               map_feature_type, ImageColumn

from protodata.utils import create_dir

import shutil
import tarfile
import numpy as np
import os
import tempfile
from six.moves import urllib
from glob import glob
import pickle
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)

CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
IMGS_PER_FILE = 10000
TRAIN_FILES = 5


class Cifar10Serialize(SerializeSettings):

    def __init__(self, data_path):
        """ See base class """
        super(Cifar10Serialize, self).__init__(data_path)
        create_dir(data_path)

        if not is_downloaded(data_path):
            logger.info('Downloading CIFAR dataset ...')
            download(data_path)

    def read(self):

        train_batches = self._get_train_batches()
        test_batch = self._get_test_batch()

        assert len(train_batches) == 5

        train_x, train_y = parse_batches(train_batches)
        test_x, test_y = parse_batches([test_batch])

        assert train_x.shape[0] == train_y.shape[0] == 50000
        assert test_x.shape[0] == test_y.shape[0] == 10000

        self.x = np.concatenate([train_x, test_x], axis=0)
        self.y = np.concatenate([train_y, test_y], axis=0)

    def _get_train_batches(self):
        files = glob(os.path.join(self.data_path, 'data_batch_*'))
        return [read_batch(f) for f in files]

    def _get_test_batch(self):
        print(self.data_path)
        print(glob(os.path.join(self.data_path, 'test_batch*')))
        test_path = glob(os.path.join(self.data_path, 'test_batch'))[0]
        return read_batch(test_path)

    def get_validation_indices(self, train_ratio, val_ratio):
        """ 
        CIFAR has already the partition between training and test.
        Only validation ratio is determined here
        """
        logger.warn('CIFAR10 is already split. Ignoring training ratio...')

        if val_ratio > 0.85:
            raise ValueError('CIFAR only support validation up to 85%')

        # Get subset for validation set
        num_validation = int(60000 * val_ratio)
        num_train = 50000 - num_validation

        # Get random indexes
        perm_train = np.random.permutation(50000)
        train = perm_train[:num_train].tolist()
        val = perm_train[num_train:].tolist(),

        # Test set already computed
        test = range(50000, 60000)
        return train, val, test

    def get_options(self):
        return {}

    def define_columns(self):
        return [
            NumericColumn('label', type=map_feature_type(np.dtype('int'))),
            ImageColumn('image', format='JPEG'),
        ]

    def build_examples(self, index):
        feature_dict = {
            'image': bytes_feature(
                self.process_image_bytes(self.x[index, ...])
            ),
            'label': int64_feature(int(self.y[index])),
        }
        return [
            tf.train.Example(
                features=tf.train.Features(feature=feature_dict)
            )
        ]


class Cifar10Settings(DataSettings):

    def __init__(self, dataset_location, image_specs=None,
                 embedding_dimensions=32, quantizer=None):
        super(Cifar10Settings, self).__init__(
            dataset_location=dataset_location,
            image_specs=image_specs,
            embedding_dimensions=embedding_dimensions,
            quantizer=quantizer)

    def tag(self):
        return 'cifar10'

    def size_per_instance(self):
        return 0.75

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
        return []


def parse_batches(batch_infos):

    n_files = len(batch_infos)
    imgs = np.zeros((IMGS_PER_FILE * n_files, 32, 32, 3))
    labels = np.zeros((IMGS_PER_FILE * n_files))

    for i, batch in enumerate(batch_infos):
        # Reshape the array to 4-dimensions and
        batch_images = batch[b'data'].reshape([-1, 3, 32, 32])
        batch_images = batch_images.transpose([0, 2, 3, 1])
        batch_labels = np.array(batch[b'labels'])

        # Append to data
        start = i * IMGS_PER_FILE
        end = start + IMGS_PER_FILE
        imgs[start:end] = batch_images
        labels[start:end] = batch_labels

    return imgs, labels


def is_downloaded(folder):
    """ Returns whether CIFAR has been downloaded """
    return os.path.isdir(folder) and len(glob(os.path.join(folder, '*'))) == 8


def read_batch(f):
    with open(f, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def download(dst):

    # Download into temp file
    # fd, down = tempfile.mkstemp()
    # urllib.request.urlretrieve(CIFAR_URL, down)

    down = '/home/dm-bw7/Downloads/cifar-10-python.tar.gz'

    # Create temporary folder where to extract subfiles
    tmp_folder = tempfile.mkdtemp()

    # Move bytes from input to output
    with tarfile.open(down) as src:
        src.extractall(path=tmp_folder)

    # Copy extracted files into destination
    extracted = os.path.join(tmp_folder, 'cifar-10-batches-py')
    files = [
        f for f in os.listdir(extracted)
        if os.path.isfile(os.path.join(extracted, f))
    ]
    for f in files:
        shutil.copyfile(os.path.join(extracted, f), os.path.join(dst, f))

    # Delete tmp file
    # os.close(fd)
    # os.remove(down)

    # shutil.rmtree(tmp_folder)