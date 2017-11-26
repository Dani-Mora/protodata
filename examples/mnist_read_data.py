from protodata.reading_ops import DataReader
from protodata.datasets import MnistSettings, Datasets
from protodata.data_ops import DataMode
from protodata.image_ops import get_image_specs
from protodata.utils import NetworkModels, get_data_location

import scipy.misc
import tensorflow as tf

import logging
logger = logging.getLogger(__name__)


tf.app.flags.DEFINE_string('data_location',
                           get_data_location(Datasets.MNIST),
                           'Path where to read datasetfrom')

tf.app.flags.DEFINE_string('batch_size',
                           5,
                           'Number of instances per batch')

tf.app.flags.DEFINE_integer('memory_factor',
                            1,
                            'Factor related to the capacity of the queue' +
                            ' (~GB). The higher this amount, the more mixed' +
                            'the data but the slower the processing time')

tf.app.flags.DEFINE_integer('reader_threads',
                            4,
                            'Number of threads to read the instances.')
FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':

    with tf.Session() as sess:

        # MNIST settings
        dataset = MnistSettings(dataset_location=FLAGS.data_location,
                                image_specs=get_image_specs(
                                    NetworkModels.MNIST,
                                    batch_size=FLAGS.batch_size,
                                    mean=[0.0, 0.0, 0.0],
                                    random_crop=False))

        # Read batches from dataset
        reader = DataReader(dataset)
        features, label = reader.read_batch(
            batch_size=FLAGS.batch_size,
            data_mode=DataMode.TRAINING,  # Whichever works here e.g. training
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
