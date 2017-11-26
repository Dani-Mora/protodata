"""

Reads the Proto files containing the Airbnb dataset. Expects several
files to be contained in the directory, identified by the set of data they
belong to and the slice of data they represent.

Note that images are returned in BGR format.

For a summary of the columns see airbnb_build_data.py.

"""

from protodata.data_ops import DataMode
from protodata.reading_ops import DataReader
from protodata.datasets import AirbnbSettings, Datasets
from protodata.image_ops import get_alexnet_specs
from protodata.utils import get_data_location

import scipy
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)


tf.app.flags.DEFINE_string('batch_size',
                           1,
                           'Number of instances per batch')

tf.app.flags.DEFINE_string('data_location',
                           # Note AIRBNB.AVAILABLE can be also
                           # generated in the notebooks
                           get_data_location(Datasets.AIRBNB_PRICE),
                           'Path where to build dataset')

tf.app.flags.DEFINE_integer('memory_factor',
                            2,
                            'Factor related to the capacity of the queue' +
                            ' (~GB). The higher this amount, the more mixed' +
                            'the data but the slower the processing time')

tf.app.flags.DEFINE_integer('reader_threads',
                            4,
                            'Number of threads to read the instances.')

FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':

    with tf.Session() as sess:

        # Airbnb settings
        dataset = AirbnbSettings(
            dataset_location=FLAGS.data_location,
            image_specs=get_alexnet_specs(FLAGS.batch_size,
                                          random_crop=True))

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
        scipy.misc.imshow(example['image'][idx])

        coord.request_stop()
        coord.join(threads)
