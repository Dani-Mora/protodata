"""

Quantization classes that ideally converts numeric targets into
categorical columns

"""

from dataio.data_ops import copy_columns, get_interval_mask
from dataio.utils import NON_SAVABLE_COL, get_logger
import abc
import tensorflow as tf

logger = get_logger(__file__)


class BaseQuantize(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, edges):
        """ Buils the quantize object.

        Args:
            edges: Edge values that separate the different bins.

        """
        if len(edges) < 1:
            raise ValueError('At least one cutting edge must be defined')

        self.edges = edges

    def num_classes(self):
        """ Returns the number of classes defined by the quantization """
        return len(self.edges) + 1

    def quantize(self, values):
        """  Converts the input target column into binned categories """
        if values.get_shape().ndims > 1:
            values = tf.squeeze(values, [1])
        return self._quantize(values)

    @abc.abstractmethod
    def _quantize(self, values):
        """ Quantizer specific method """

    @abc.abstractmethod
    def target_type(self):
        """ Returns the target type of the resulting target column.
        By default it is an integer """


class Quantize(BaseQuantize):

    """
        Quantizes input numerical values into categories.
        A value x is put into bin 'i'

            when:
                bins[i] <= x < bins[i+1]
            With special case for i == 0:
                x < bins[0]
            and special case for i == len(bins)-1:
                x >= bins[-1]

        This is only valid when the input tensor is a 1D tensor.
        Otherwise it will raise an error.

        This quantizer assignes each input value into a specific class.

    """

    def __init__(self, edges, batch_size):
        super(Quantize, self).__init__(edges)
        self.batch_size = batch_size

    def _quantize(self, values):
        """ See base class """

        if len(values.get_shape()) != 1:
            raise ValueError("Input tensor must be a 1D Tensor")

        logger.info("Performing basic quantization using edges {}"
                    .format(self.edges))

        # Create new variable for quantized values
        new_values = _auxiliar_variable(name='quantize_data',
                                        shape=self.batch_size,
                                        dtype=tf.float32,
                                        initializer=tf.zeros_initializer())

        # Iterate over edges defined
        for i in range(len(self.edges)):

            if i == 0:
                # Exclusively lower than bins[0]
                mask = tf.less(values, self.edges[i])
            else:
                # bins[i-1] <= x < bins[i]
                mask = get_interval_mask(self.edges[i - 1],
                                         self.edges[i],
                                         values)

            # Add current bin category
            new_values = assign(new_values, mask, i)

            # Add extra category for values exclusively >= to bins[-1]
            if i == len(self.edges) - 1:
                extra_mask = tf.greater_equal(values, self.edges[i])
                new_values = assign(new_values, extra_mask, i + 1)

        return tf.cast(new_values, dtype=tf.int32)

    def target_type(self):
        """ Returns the target type of the resulting target column.
        By default it is an integer """
        return tf.int32


class SoftQuantize(BaseQuantize):

    """

    This quantizer resembles the Quantize class but generates a
    one-hot enconding instead. This encoding is built in a way that the
    score of a category is proportional to the distance between the bin
    center and the given value. A minimum probability mass can be given to the
    true value to ensure that it receives enough probability.
    It assumes minimum input number is 0.

    """

    def __init__(self, edges, on_val=0.40, factor=0.25, delta=1e-3):
        """ Initializes a soft quantizer.
        Args:
            edges: Edges defining the different intervals used to bin
                the input values. First category are values below first
                edge and last one are the values above the last edge.
            on_val: Minimum probability of the true interval. The remaining
                probability is proportionally distributed among all
                (true one included) classes.
            factor: It is the fraction of the last edge used to compute the
                distance to the last bin center.
            delta: Delta added to the differences between input values and
                bin centers to avoid zero divisions.
        """
        super(SoftQuantize, self).__init__(edges)

        if on_val <= 0 or on_val >= 1:
            raise ValueError("Groundtruth probability must be in (0,1)")

        if factor <= 0 or on_val >= 1:
            raise ValueError("Last edge rate must be in (0,1)")

        self.on_val = on_val
        self.factor = factor
        self.delta = delta

    def _quantize(self, values):
        """ See base class """
        if len(values.get_shape()) != 1:
            raise ValueError("Input tensor must be a 1D Tensor")

        logger.info("Soft quantization with edges {}" .format(self.edges) +
                    "Minimum probability {}".format(self.on_val) +
                    " and factor {}".format(self.factor))

        off = 1.0 - self.on_val

        # Get bin centers
        new_edges = [0] + self.edges
        centers = [get_center(new_edges[i], new_edges[i + 1])
                   for i in range(0, len(new_edges) - 1)]

        # Add center for last bin
        centers = centers + [self.edges[-1] + self.edges[-1] * self.factor]

        # Get difference between bins and values
        diffs = [tf.abs(tf.add(values, -c)) for c in centers]
        diffs = tf.stack(diffs, axis=1)
        # Add a small delta to avoid differences of 0
        diffs = tf.add(diffs, tf.ones(shape=tf.shape(diffs)) * self.delta)

        # Return inverse so closer distance has higher probability
        inv_dist = tf.reciprocal(diffs)

        # Row-based sum: create tiled version
        summed = tf.reduce_sum(inv_dist, axis=1)
        summed = copy_columns(summed, self.num_classes())

        # Get soft probabilities
        weights = tf.div(inv_dist, summed)
        weights = tf.multiply(weights, tf.constant(off))

        # Set base probability using groundtruth bins
        one_hot = tf.one_hot(tf.argmin(diffs, axis=1),
                             on_value=self.on_val,
                             off_value=0.0,
                             depth=self.num_classes())

        # Add groundtruth and soft labelling
        return tf.add(one_hot, weights)

    def target_type(self):
        """ Returns the target type of the resulting target column.
        By default it is an integer """
        return tf.float32


def get_center(v1, v2):
    """ Returns the center point between both inputs """
    return (v1 + v2) / 2.0


def assign(new_tensor, mask, value):
    """ Assigns value to those positions given by mask """
    # Positions where to assign the value
    indices = tf.squeeze(tf.where(mask), [1])
    # Create tensor with repeated value
    to_assign = tf.cast(tf.ones(shape=tf.shape(indices)) * value, tf.float32)
    # Assign value into the corresponding positions
    return tf.scatter_update(new_tensor, indices, to_assign)


def _auxiliar_variable(name, shape, dtype, initializer):
    """ Creates or retrieves an existing variable that won't be stored
    in the checkpoints. This variable is not trainable"""
    collections = set([tf.GraphKeys.GLOBAL_VARIABLES] + [NON_SAVABLE_COL])
    return tf.get_variable(name=name,
                           shape=shape,
                           dtype=dtype,
                           initializer=initializer,
                           collections=collections,
                           trainable=False)
