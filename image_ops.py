from protodata.utils import NetworkModels

import numpy as np
import tensorflow as tf
import os


""" Image manipulation functions and classes """


# Inspired in class from GitHub user etheron
# Found in: https://github.com/ethereon/caffe-tensorflow/blob/master/examples/imagenet/models/helper.py # noqa
class DataSpec(object):

    """ Input data specifications for an image model """

    def __init__(self, batch_size=128, scale_size=256, crop_size=227,
                 isotropic=True, channels=3, mean=None, bgr=True,
                 random_crop=True):
        """ Initializes the image specs
        Args:
            batch_size: Recommended batch size for the model

            scale_size: Size image should be scaled to first when
                adjusting its size

            isotropic: Whether the model expects the rescaling to
                be isotropic (respect ratio)

            crop_size: Side size of the square crop expected the
                image should be resized to after scaling

            channels: Number of channels of the image

            mean: The mean to be subtracted from each image. By default,
                he per-channel ImageNet mean. The values below are ordered
                BGR, as many Caffe models are trained in this order. Some
                of the earlier models (like AlexNet) used a spatial three-
                channeled mean. However, using just the per-channel mean
                values instead doesn't affect things too much.

            random_crop: Whether to crop images around using same offset along
                each dimension or using a slightly random crop around center.
        """
        self.batch_size = batch_size
        self.scale_size = scale_size
        self.isotropic = isotropic
        self.crop_size = crop_size
        self.channels = channels
        if mean is None:
            self.mean = np.array([104., 117., 124.]) \
                if bgr else np.array([124., 104., 117.])
        else:
            self.mean = mean
        self.expects_bgr = bgr
        self.random_crop = random_crop

    def from_config(self, section):
        """ Reads the image configuration from a configuration parser """
        self.batch_size = section.getint('batch_size')
        self.scale_size = section.getint('scale_size')
        self.crop_size = section.getint('crop_size')
        self.isotropic = section.getboolean('isotropic')
        self.channels = section.getint('channels')
        self.mean = [section.getfloat('mean_' + str(i))
                     for i in range(self.channels)]
        self.expects_bgr = section.getboolean('expects_bgr')

    def to_config(self, config):
        """ Stores the image characteristics into a configparser section """
        config['batch_size'] = self.batch_size
        config['scale_size'] = self.scale_size
        config['crop_size'] = self.crop_size
        config['isotropic'] = self.isotropic
        config['channels'] = self.channels
        for i in range(self.channels):
            config['mean_' + str(i)] = self.mean[i]
        config['expects_bgr'] = self.expects_bgr

    def decode(self, img_content, is_jpeg):
        """ Decodes image from a TF proto example into a Tensor
        Args:
            img_content: image protobuf content
            is_jpeg: whether image is in jpeg format (True) or in png (False)
        Return:
            Tensor containing the original image
        """
        return decode_image(img_content,
                            channels=self.channels,
                            is_jpeg=is_jpeg,
                            expects_bgr=self.expects_bgr)

    def adjust(self, img):
        """ Resizes ad crops image according to settings
        Args:
            img: Image tensor
        Return:
            Reshaped image tensor
        """
        return adjust_image(img=img,
                            scale=self.scale_size,
                            crop=self.crop_size,
                            isotropic=self.isotropic,
                            mean=self.mean,
                            random_crop=self.random_crop)

    def decode_and_adjust(self, img_content, is_jpeg):
        """ Decodes image from a TF proto example.
        Resizes and crops the image according to the settings
        Args:
            img_content: image protobuf content
            is_jpeg: whether image is in jpeg format (True) or in png (False)
        Returns:
            Tensor containing the processed image
        """
        decoded = self.decode(img_content, is_jpeg)
        return adjust_image(img=decoded,
                            scale=self.scale_size,
                            crop=self.crop_size,
                            isotropic=self.isotropic,
                            mean=self.mean,
                            random_crop=self.random_crop)


def get_alexnet_specs(batch_size, mean=None, random_crop=True):
    """ Gets Alexnet image data characteristics.
    If mean is None, uses Alexnet default one (not centered around zero) """
    return DataSpec(batch_size=batch_size,
                    scale_size=256,
                    crop_size=227,
                    isotropic=False,
                    mean=mean,
                    random_crop=random_crop)


def get_vgg_specs(batch_size, mean=None, random_crop=True):
    """ Gets Alexnet image data characteristics.
    If mean is None, will use Alexnet default one (not centered around zero)
    """
    return DataSpec(batch_size=batch_size,
                    scale_size=256,
                    crop_size=224,
                    isotropic=True,
                    mean=mean,
                    random_crop=random_crop)


def get_image_specs(network, batch_size, mean=None, random_crop=True):
    """ Return the image specs according to the given network"""
    if network == NetworkModels.ALEXNET:
        return get_alexnet_specs(batch_size, mean, random_crop)
    elif network == NetworkModels.VGG:
        return get_vgg_specs(batch_size, mean, random_crop)
    elif network == NetworkModels.MNIST:
        return DataSpec(batch_size=batch_size,
                        scale_size=28,
                        crop_size=28,
                        isotropic=True,
                        mean=mean,
                        random_crop=random_crop)
    else:
        raise ValueError('Invalid network model %s' % network)


def process_image(img_path, coder):
    """ Process an image for being encoded into a TFRecord file """
    # Read image data from path
    with tf.gfile.FastGFile(img_path, 'rb') as f:
        image_data = f.read()

    # Convert to PNG if needed
    if os.path.splitext(os.path.basename(img_path)) == '.png':
        image_data = coder.png_to_jpeg(image_data)

    # Decode JPEG
    image = coder.decode_jpeg(image_data)
    return image_data, image.shape[0], image.shape[1]


def decode_image(img, channels, is_jpeg=True, expects_bgr=False):
    """ Decodes image according to format and reverses it, if requested """
    if is_jpeg:
        decoded = tf.image.decode_jpeg(img, channels=channels)
    else:
        decoded = tf.image.decode_png(img, channels=channels)

    if expects_bgr:
        # Convert from RGB channel ordering to BGR. This is important when
        # loading from Caffe models, since they use Opencv (BGR) for images
        decoded = tf.reverse(decoded, axis=[-1])

    return decoded


def adjust_image(img, scale, isotropic, crop, mean, random_crop=True):
    """ Crops, scales, and normalizes the given image.
    Args:
        scale: The image wil be first scaled to this size.
            If isotropic is true, the smaller side is rescaled to this,
            preserving the aspect ratio.
        crop: After scaling, a central crop of this size is taken.
        mean: Subtracted from the image
        random_crop: Whether to crop using around offset around the center.
            If False, the same offset is used in both sides of the dimensions.
    Returns:
        Processed image tensor
    """

    def get_random(max_val):
        """ Returns a random integer in interval [0, max_val) """
        return tf.random_uniform(shape=(),
                                 minval=0,
                                 maxval=max_val,
                                 dtype=tf.int32)

    # Rescale
    if isotropic:
        img_shape = tf.to_float(tf.shape(img)[:2])
        min_length = tf.minimum(img_shape[0], img_shape[1])
        new_shape = tf.to_int32((scale / min_length) * img_shape)
    else:
        new_shape = tf.stack([scale, scale])

    # Resize accordingly
    img = tf.image.resize_images(img, [new_shape[0], new_shape[1]])

    # Crop according to settings
    if random_crop:
        # Crop using random offset in X and Y
        offset = (new_shape - crop)
        offset_y, offset_x = get_random(offset[0]), get_random(offset[1])
    else:
        # Crop around center
        offset = (new_shape - crop) / 2
        offset_y, offset_x = offset[0], offset[1]

    # Crop using computed offset
    img = tf.image.crop_to_bounding_box(img,
                                        tf.cast(offset_y, tf.int32),
                                        tf.cast(offset_x, tf.int32),
                                        crop,
                                        crop)

    # Mean subtraction. This is used so Places365 works as expected
    return tf.to_float(img) - mean

    # TODO: If no mean provided we could use image standardization
    # Per image normalization
    # return tf.image.per_image_standardization(img)


def create_extension_mask(paths):
    def is_jpeg(path):
        extension = os.path.splitext(path)[-1].lower()
        if extension in ('.jpg', '.jpeg'):
            return True
        if extension != '.png':
            raise ValueError('Unsupported image format: {}'.format(extension))
        return False

    return [is_jpeg(p) for p in paths]


# Adapted from Imagenet tutorial
# Source: https://raw.githubusercontent.com/tensorflow/models/master/inception/inception/data/build_imagenet_data.py # noqa

class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image,
                                                 format='rgb',
                                                 quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data,
                                                 channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def finalize(self):
        self._sess.close()
