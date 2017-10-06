from protodata.datasets import TitanicSettings
from protodata.datasets import Datasets
from protodata.reading_ops import DataReader
from protodata.data_ops import DataMode
from protodata.utils import get_data_location, get_logger

import tensorflow as tf


tf.app.flags.DEFINE_string('data_location',
                           get_data_location(Datasets.TITANIC),
                           'Path where to build dataset')

tf.app.flags.DEFINE_string('batch_size',
                           5,
                           'Number of instances per batch')

tf.app.flags.DEFINE_integer('memory_factor',
                            1,
                            'Factor related to the capacity of the queue' +
                            ' (~GB). The higher this amount, the more mixed' +
                            'the data but the slower the processing time')

tf.app.flags.DEFINE_integer('reader_threads',
                            1,
                            'Number of threads to read the instances.')

FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':

    logger = get_logger(__name__)

    with tf.Session() as sess:

        # Boston settings
        dataset = TitanicSettings(dataset_location=FLAGS.data_location)

        # Read batches from dataset
        reader = DataReader(dataset)
        features, label = reader.read_batch(
            batch_size=FLAGS.batch_size,
            data_mode=DataMode.TRAINING,  # Use whatever here, e.g. training
            memory_factor=FLAGS.memory_factor,
            reader_threads=FLAGS.reader_threads,
            train_mode=False
        )

        # Initi all vars
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Define coordinator to handle all threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        example, l = sess.run([features, label])

        # Print first instance in batch
        idx = 0
        example_idx = {k: v[idx] for k, v in example.items()}
        logger.info('Example info: {}'.format(example_idx))
        logger.info('Label: ' + str(l[idx]))

        coord.request_stop()
        coord.join(threads)
