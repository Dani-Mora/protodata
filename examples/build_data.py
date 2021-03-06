from protodata.serialization_ops import DataSerializer
from protodata import datasets
from protodata.utils import get_data_location, get_tmp_data_location

import tensorflow as tf
import logging

logging.basicConfig(level=logging.DEBUG)

dataset = datasets.Datasets.MOTOR
serialize_fn = datasets.MotorSerialize
folded = True

tf.app.flags.DEFINE_string('raw_data_location',
                           get_tmp_data_location(dataset),
                           'Path where to extract raw data')

tf.app.flags.DEFINE_string('data_location',
                           get_data_location(dataset, folded=folded),
                           'Path where to build dataset')

tf.app.flags.DEFINE_float('train_ratio', 0.80, 'Ratio of training instances')

tf.app.flags.DEFINE_float('n_folds', 10,
                          'Number of folds to use')

tf.app.flags.DEFINE_integer('num_threads',
                            1,
                            'Number of threads to use.')

tf.app.flags.DEFINE_integer('test_shards',
                            1,
                            'Number of TFRecord files for testing.')

tf.app.flags.DEFINE_float('files_per_fold',
                          1,
                          'Number of files for each fold')

FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':

    # Configuration for extraction
    settings = serialize_fn(data_path=FLAGS.raw_data_location)

    # Save to TFRecord
    serializer = DataSerializer(settings)
    serializer.serialize_folds(output_folder=FLAGS.data_location,
                               train_ratio=FLAGS.train_ratio,
                               n_folds=FLAGS.n_folds,
                               num_threads=FLAGS.num_threads,
                               test_shards=FLAGS.test_shards,
                               files_per_fold=FLAGS.files_per_fold)
