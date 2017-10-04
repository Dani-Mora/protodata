"""

Serialization of the MNIST dataset into Example protos.

Creates a categorical column for each pixel in the image (784 columns)
plus the usual image and label columns

"""

from protodata.serialization_ops import DataSerializer
from protodata.datasets import Datasets
from protodata.datasets import SonarSerialize
from protodata.utils import get_data_location, get_tmp_data_location

import tensorflow as tf

tf.app.flags.DEFINE_string('raw_data_location',
                           get_tmp_data_location(Datasets.SONAR),
                           'Path where to extract raw data')

tf.app.flags.DEFINE_string('data_location',
                           get_data_location(Datasets.SONAR),
                           'Path where to build dataset')

# Dataset settings
tf.app.flags.DEFINE_float('train_ratio', 0.80,
                          'Ratio of training instances.')

tf.app.flags.DEFINE_float('val_ratio',
                          0.10,
                          'Ratio of validation instances.')

tf.app.flags.DEFINE_integer('train_shards',
                            1,
                            'Number of shards in training TFRecord files.')

tf.app.flags.DEFINE_integer('validation_shards',
                            1,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('test_shards',
                            1,
                            'Number of shards in testing TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads',
                            1,
                            'Number of threads to use.')

FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':

    # Configuration for extraction
    settings = SonarSerialize(data_path=FLAGS.raw_data_location)

    # Save to TFRecord
    serializer = DataSerializer(settings)
    serializer.serialize(output_folder=FLAGS.data_location,
                         train_ratio=FLAGS.train_ratio,
                         val_ratio=FLAGS.val_ratio,
                         num_threads=FLAGS.num_threads,
                         train_shards=FLAGS.train_shards,
                         val_shards=FLAGS.validation_shards,
                         test_shards=FLAGS.test_shards)
