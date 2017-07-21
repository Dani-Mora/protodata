
""" Test of quantize classes """

from dataio.quantize import Quantize, SoftQuantize
import numpy as np
import tensorflow as tf
import logging
import unittest


class QuantizeBaseTestCase(unittest.TestCase):

    def setUp(self):
        self.values = np.asarray([23, 50, 42, 10, 65, 80])
        self.edges = [25, 50, 75]
        self.num_classes = len(self.edges) + 1
        self.batch_size = len(self.values)

        self.logger = logging.getLogger(__name__)

        # Define Tensorflow placeholder
        self.pl = tf.placeholder(tf.float32, shape=[None])

    def test_soft_quantize(self):

        on_val = 0.35
        factor = 0.25

        # Smooth quantize
        sq = SoftQuantize(edges=self.edges, on_val=on_val, factor=factor)
        sq_out = sq.quantize(self.pl)

        # Send to execution
        with tf.Session() as sess:

            # Init all variables
            sess.run(tf.global_variables_initializer())

            # Log results
            result = sess.run([sq_out], feed_dict={self.pl: self.values})
            self.logger.info("Smooth quantizing {} with edges {}"
                             .format(self.values, self.edges))
            self.logger.info("Quantizing arameters ({}, {}): \n{}"
                             .format(on_val, factor, result[0]))

            self.assertTrue(result[0].shape ==
                            (self.batch_size, self.num_classes))

    def test_base_quantize(self):

        # Basic quantize
        bq = Quantize(edges=self.edges, batch_size=self.batch_size)
        bq_out = bq.quantize(self.pl)

        # Send to execution
        with tf.Session() as sess:

            # Init all variables
            sess.run(tf.global_variables_initializer())

            # Log results
            result = sess.run([bq_out], feed_dict={self.pl: self.values})
            self.logger.info("Smooth quantizing {} with edges {}: {}"
                             .format(self.values, self.edges, result[0]))

            self.assertTrue(np.array_equal(result[0],
                                           np.asarray([0, 2, 1, 0, 2, 3])))


if __name__ == '__main__':
    unittest.main()
