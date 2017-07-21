import dataio.data_ops as do
from dataio.utils import FileNotFound, load_pickle, get_logger
from dataio.quantize import BaseQuantize

import abc
from abc import ABCMeta
import os
import tensorflow as tf

logger = get_logger(__name__)


""" File containing code to batch serialized data.
    The data is expected to be stored in protobuf Tensorflow
    records using the 'serialization_ops.py' functions. """


class DataReader(object):

    def __init__(self, settings):
        """ Args:
            settings: Dataset settings
        """
        if not isinstance(settings, DataSettings):
            raise TypeError('Attribute must be a subclass of DataSettings')
        self.settings = settings
        self._image_tag = 'image'
        self._jpeg_tag = 'jpeg'
        self._png_tag = 'png'

    def read_batch(self, batch_size, data_mode, memory_factor, reader_threads,
                   train_mode=True, shuffle=True):
        """ Returns the dequeued serialized TFRecords from the dataset.
        Args:
            batch_size: Estimated batch size.
            data_mode: Whether we are reading training, validation or
                testing data
            memory_factor: Factor related to memory usage for enqueuing
                (~GB to use) at maximum.
            reader_threads: Number of parallel readers.
            train_mode: Whether to infinitely process batches (True)
                or finish once first epoch ends (False)
            shuffle: Whether to shuffle examples

        This function reads unlimited batches for training
        (train_mode == True) and a single epoch for other modes
        """

        logger.info('Reading batches from %s' % data_mode)

        logger.info('Using batch size of %d, memory of ~%d GB and %d threads'
                    % (batch_size, memory_factor, reader_threads))

        if self.settings.image_column() is not None \
                and self.settings.image_specs is None:
            raise ValueError('Image field provided but no image specs found')
        if data_mode not in [do.DataMode.TRAINING,
                             do.DataMode.VALIDATION,
                             do.DataMode.TEST]:
            raise ValueError('Invalid data mode %s' % data_mode)

        has_image = self.settings.image_column()

        with tf.name_scope('batch_processing'):

            records = self.settings.get_files(data_mode)

            # Define queue for the files
            if train_mode:
                file_queue = tf.train.string_input_producer(
                    records,
                    shuffle=shuffle,
                    capacity=memory_factor,
                    num_epochs=None
                )
            else:
                file_queue = tf.train.string_input_producer(records,
                                                            shuffle=shuffle,
                                                            num_epochs=1)

            # Read records
            reader = tf.TFRecordReader()
            _, example = reader.read(file_queue)

            # Decode image data if required
            if has_image:
                to_batch = [example, self.decode_images(example)]
            else:
                to_batch = [example]

            # Group into batches
            batched = self.batch_data(data=to_batch,
                                      batch_size=batch_size,
                                      memory_factor=memory_factor,
                                      reader_threads=reader_threads)

            # Decode image data if required
            if has_image:
                batched_examples, batched_images = batched
            else:
                batched_examples = batched

            # The serialized example is converted back to actual values
            # Important to use parse_example after batching for Sparse Data
            # Doc: goo.gl/Vm19xf
            parsed = tf.parse_example(batched_examples,
                                      self.settings.get_feature_dictionary())

            base_features, labels = self.separate_target(parsed, batch_size)

            # Decode image data if required
            if has_image:
                base_features.update({self._image_tag: batched_images})

        return base_features, tf.cast(labels, self.settings.target_type())

    def batch_data(self, data, memory_factor, batch_size, reader_threads):
        """ Groups data into random batches """
        # Minimum dequeue examples as examples to fit in memory specified
        # Capacity contains 3 more batches than the minimum dequeue amount
        size_per_instance = self.settings.size_per_instance()
        min_after_deq = int((1024 * memory_factor) / float(size_per_instance))
        capacity = min_after_deq + 3 * batch_size

        batched = tf.train.shuffle_batch(
            data,
            batch_size=batch_size,
            num_threads=reader_threads,
            allow_smaller_final_batch=False,  # Don't let smaller batches
            capacity=capacity,  # Pre-fetching capacity
            min_after_dequeue=min_after_deq  # Values needed before dequeue
        )
        return batched

    def separate_target(self, parsed, batch_size):
        """ Separates features from the target class """
        base_features = {f: v for f, v in parsed.items()
                         if f != self.settings.target_class()}
        labels = parsed[self.settings.target_class()]
        return base_features, self.settings.parse_labels(labels, batch_size)

    def decode_images(self, example):
        """ Decode image content and resizes it, if provided """
        # Parse only the image data
        image_column = self.settings.image_column()
        image_dict = {image_column.name: image_column.get_feature()}
        image_data = tf.parse_single_example(example, image_dict)

        # Extract image feature from list of features
        image_content = image_data[image_column.name]

        # Resize and decode image accordingly
        return self.settings.image_specs.decode_and_adjust(
            image_content,
            is_jpeg=self.get_image_format() == self._jpeg_tag)

    def get_image_format(self):
        """ Returns the format of the images in the dataset """
        if self.settings.image_column() is None:
            return None
        else:
            if self.settings.image_format().lower() \
                    in ['jpeg', 'jpg', '.jpeg', '.jpg']:
                return self._jpeg_tag
            elif self.settings.image_format().lower() in ['png', '.png']:
                return self._png_tag
            else:
                raise ValueError('Not supported image format %s'
                                 % self.settings.image_format())


class DataSettings(object):

    """" Class that gathers metadata and content of a dataset """

    __metaclass__ = ABCMeta

    def __init__(self, dataset_location, image_specs=None,
                 embedding_dimensions=32, quantizer=None):
        """
        Args:
            dataset_location: Folder where dataset is stored
            image_specs: Measures to use for the images.
                Set to None if no images to be loaded.
            embedding_dimensions: Number of dimensions deep columns
                will be embedded in.
            quantizer: Quantizer class to categorize the target values.
                Set to None to disable.
        """
        if not os.path.exists(dataset_location):
            raise FileNotFound('No folder exists at location %s'
                               % dataset_location)

        self.dataset_location = dataset_location
        self.serialize_options = load_pickle(os.path.join(dataset_location,
                                                          'metadata.dat'))
        self.image_specs = image_specs

        # Store columns as dictionary
        self.columns = {c.name: c for c in self.serialize_options['columns']}

        # Identify image content. We only accept one ImageColumn
        img_cols = [i for i in self.serialize_options['columns']
                    if isinstance(i, do.ImageColumn)]

        if len(img_cols) > 1:
            raise ValueError('Dataset must only contain one single image')

        self.image_col = None if not img_cols else img_cols[-1]
        self.image_form = None if not img_cols else img_cols[-1].format

        self.embedding_dims = embedding_dimensions

        if quantizer is not None:
            if not isinstance(quantizer, BaseQuantize):
                raise ValueError("Quantizer must be a quantizer object")
        self.quantizer = quantizer

    def target_type(self):
        """ Returns the type of data of the target """
        return self._target_type() \
            if self.quantizer is None else self.quantizer.target_type()

    def get_num_classes(self):
        """ Returns the size of the target column """
        return self._get_num_classes() \
            if self.quantizer is None else self.quantizer.num_classes()

    @abc.abstractmethod
    def size_per_instance(self):
        """ Approximate size per instance """

    @abc.abstractmethod
    def target_class(self):
        """ Name of the column to predict """

    @abc.abstractmethod
    def _target_type(self):
        """ Default type of the column to predict """

    @abc.abstractmethod
    def _get_num_classes(self):
        """ Default size of the outputs of the data """

    @abc.abstractmethod
    def select_wide_cols(self):
        """ Selects wide columns for the corresponding dataset """

    @abc.abstractmethod
    def select_deep_cols(self):
        """ Selects wide columns for the corresponding dataset """

    def image_column(self):
        """ ImageColumn corresponding to the containing image content.
        None if no image in the dataset """
        return self.image_col

    def image_format(self):
        """ Format of the image content. None if no image content.
        Supported: 'jpeg', 'png' """
        return self.image_form

    def parse_labels(self, labels, batch_size):
        """ Function that parses labels. By default does nothing.
        Override if needed """
        labels = self._parse_labels(labels, batch_size)
        if self.quantizer is not None:
            return self.quantizer.quantize(labels)
        else:
            return labels

    def _parse_labels(self, labels, batch_size):
        """ Function that parses labels into the proper format.
        Override if needed """
        return labels

    def get_feature_dictionary(self):
        """ Returns the feature dictionary to parse non-image
        columns from the TFRecords
        Returns
            features: Dictionary of features metadata
        """
        f_dict = {}
        for k, v in self.columns.items():
            if not isinstance(v, do.ImageColumn):
                f_dict[k] = v.get_feature()
        return f_dict

    def get_files(self, data_mode):
        """ Returns the TF records for the dataset subset """
        # Data is read in a specific pattern
        location = self.dataset_location
        # Data read using pattern
        tf_record_pattern = do.get_filename_pattern(location, data_mode)
        data_files = tf.gfile.Glob(tf_record_pattern)
        if not data_files:
            raise FileNotFound('No records files found in %s and %s subset'
                               % (location, data_mode))
        return data_files

    def get_image_shape(self):
        return [
            self.image_specs.crop_size,
            self.image_specs.crop_size,
            self.image_specs.channels
        ]

    def _get_columns(self, columns, excluded_cols=[]):
        """ Returns the set of input columns that are not part of
        exclusion input list """
        selected = []
        for i in columns:
            # Get representative column name
            if hasattr(i, 'source_column'):
                col_name = i.source_column.column_name
            elif hasattr(i, 'name'):
                col_name = i.name
            else:
                raise ValueError('Unknown column type {}'.format(i))

            if col_name not in excluded_cols:
                selected.append(i)

        return selected

    def get_deep_columns(self, excluded_cols=[]):
        """ Returns the set of deep columns excluding the given
         ones and the target one """
        return self._get_columns(self.select_deep_cols(),
                                 excluded_cols + [self.target_class()])

    def get_wide_columns(self, excluded_cols=[]):
        """ Returns the set of wide columns excluding the given
         ones and the target one"""
        return self._get_columns(self.select_wide_cols(),
                                 excluded_cols + [self.target_class()])

    def get_image_column(self):
        """ Returns the corresponding image column from the dataset """
        if self.image_column() is None:
            raise NotImplementedError("Image column not found in dataset")
        return [self.image_column().to_column(self.get_image_shape())]
