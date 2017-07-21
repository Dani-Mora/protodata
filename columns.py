import tensorflow as tf
import collections


class _ImageColumn(tf.contrib.layers.feature_column._FeatureColumn,
                   collections.namedtuple("_ImageColumn",
                                          ['column_name',
                                           'height',
                                           'width',
                                           'channels',
                                           'dtype'])):

    """Represents a feature column built from raw images. """

    def __new__(cls,
                column_name,
                height,
                width,
                channels,
                dtype=tf.float32):
        return super(_ImageColumn, cls).__new__(cls,
                                                column_name,
                                                height,
                                                width,
                                                channels,
                                                dtype)

    @property
    def name(self):
        return self.column_name

    @property
    def config(self):
        return {self.column_name: tf.FixedLenFeature(self.dtype)}

    @property
    def key(self):
        """Returns a string which will be used as a key when we do sorting."""
        return "{}".format(self)

    @property
    def image_dimensions(self):
        return [self.height, self.width, self.channels]

    # pylint: disable=unused-argument
    def weight_tensor(self, input_tensor):
        """Returns the weight tensor from the transformed input_tensor."""
        return None

    def insert_transformed_feature(self, columns_to_tensors):
        """ Transforms feature column into tensor """
        # Name has been reused for DNN but note this relates to CNNs
        reshaped = tf.reshape(columns_to_tensors[self.name],
                              [-1] + list(self.image_dimensions))
        columns_to_tensors[self] = tf.cast(reshaped, dtype=tf.float32)

    def _to_dnn_input_layer(self,
                            input_tensor,
                            weight_collections=None,
                            trainable=True,
                            output_rank=None):
        return input_tensor


def create_image_column(name, dims, dtype=tf.float32):
    """ Creates an instance of an ImageColumn
        name: Name of the feature column
        dims: Image dimensions [Height, Width, Channels]
        dtype: Type of the image content
    """
    height, width, channels = dims
    return _ImageColumn(name, height, width, channels, dtype)
