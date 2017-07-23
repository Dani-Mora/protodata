"""

Serialization of the Boston dataset into Example protos.

For more information about the data check scikit_dataset.py

"""

from protodata.serialization_ops import DataSerializer
from protodata.datasets import BostonSerialize, Datasets
from protodata.utils import get_data_location

import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

tf.app.flags.DEFINE_float('train_ratio', 0.80,
                          'Ratio of training instances. ' +
                          'The ratio is applied to rooms')

tf.app.flags.DEFINE_float('val_ratio', 0.10,
                          'Ratio of validation instances. ' +
                          'The ratio is applied to rooms')

# Serialization parameters
tf.app.flags.DEFINE_string('data_location',
                           get_data_location(Datasets.BOSTON),
                           'Path where to store dataset')

tf.app.flags.DEFINE_integer('train_shards',
                            4,
                            'Number of shards in training TFRecord files.')

tf.app.flags.DEFINE_integer('validation_shards',
                            2,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('test_shards',
                            2,
                            'Number of shards in testing TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads',
                            2,
                            'Number of threads to preprocess the images.')

FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':

    # Configuration for extraction
    settings = BostonSerialize()

    # Save to TFRecord
    serializer = DataSerializer(settings)

    # Serialize data
    serializer.serialize(output_folder=FLAGS.data_location,
                         train_ratio=FLAGS.train_ratio,
                         val_ratio=FLAGS.val_ratio,
                         num_threads=FLAGS.num_threads,
                         train_shards=FLAGS.train_shards,
                         val_shards=FLAGS.validation_shards,
                         test_shards=FLAGS.test_shards)
