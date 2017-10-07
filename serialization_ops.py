import protodata.data_ops as do
from protodata.utils import save_pickle, create_dir, download_file, get_logger
from protodata.image_ops import ImageCoder, process_image

import threading
from datetime import datetime
import scipy.misc
import tempfile
import os
from abc import ABCMeta
import abc
import numpy as np
import tensorflow as tf

logger = get_logger(__name__)


""" File containing code to generate TFRecords from datasets.
    This records are protobuf serialization versions of the data
    and are the recommended standard for Tensorflow
    Inspired from goo.gl/fZyRit """


class DataSerializer(object):

    """ Serializes a dataset into TFRecords """

    def __init__(self, serialize_settings):
        if not isinstance(serialize_settings, SerializeSettings):
            raise TypeError('Attribute must be subclass of SerializeSettings')
        self.settings = serialize_settings

    def serialize(self,
                  output_folder,
                  train_ratio,
                  val_ratio,
                  num_threads,
                  train_shards,
                  val_shards,
                  test_shards):
        """ Serializes the data into a Tensorflow recommended
        Example proto format
        Args:
            output_folder: Output folder for the record files.
            train_ratio: Ratio of instances in the training data.
                If original dataset already split, this is not used.
            val_ratio: Ratio of instances in the validation data.
            num_threads: Threads to use.
            train_shards: Number of files the training set will be split in.
                Must be divisible by the number of threads.
            val_shards: Number of slices the validation set will be split in.
                Must be divisible by the number of threads.
            test_shards: Number of slices the testing set will be split in.
                Must be divisible by the number of threads.
        """

        logger.info("Trying to create dataset into %s" % output_folder)

        if train_ratio > 1.0 or train_ratio < 0.0:
            raise ValueError('Training ratio must be in interval [0, 1]')
        if val_ratio > 1.0 or val_ratio < 0.0:
            raise ValueError('Validation ratio must be in interval [0, 1]')
        if train_ratio + val_ratio >= 1.0:
            raise ValueError('Training and validation ratio exceed 1')
        if os.path.exists(output_folder):
            raise ValueError('Dataset already exists!')

        create_dir(output_folder)

        # Read dataset
        self.settings.initialize()

        # Split according to validation preferences
        logger.info('Splitting into training and validation')
        train, val, test = \
            self.settings.get_validation_indices(train_ratio, val_ratio)

        # Create training files
        self._store_dataset(train,
                            output_folder,
                            train_shards,
                            num_threads,
                            do.DataMode.TRAINING)

        # Create validation files
        self._store_dataset(val,
                            output_folder,
                            val_shards,
                            num_threads,
                            do.DataMode.VALIDATION)

        # Create test files
        self._store_dataset(test,
                            output_folder,
                            test_shards,
                            num_threads,
                            do.DataMode.TEST)

        # Store settings
        self._store_options(output_folder,
                            train_shards,
                            val_shards,
                            num_threads)

        # Free resources, if any
        self.settings.finalize()

    def serialize_folds(self,
                        output_folder,
                        n_folds,
                        num_threads,
                        files_per_fold=1):
        """ Serializes the data into a Tensorflow recommended
        Example proto format using N folds. Each fold has its own
        folder with a certain amount of files.
        Args:
            output_folder: Output folder for the record files.
            n_folds: Ratio of instances in the training data.
                If original dataset already split, this is not used.
            num_threads: Number of threads to use.
            instances_per_file: Number of instances contained in each file.
        """
        logger.info("Trying to create folded dataset into %s" % output_folder)

        if os.path.exists(output_folder):
            raise ValueError('Dataset already exists!')

        create_dir(output_folder)

        # Read dataset
        self.settings.initialize()

        # Compute fold statistics
        n_instances = self.settings.get_instance_num()
        idx_per_fold = np.array_split(range(n_instances), n_folds)

        for fold in range(n_folds):
            self._store_dataset(idx_per_fold[fold],
                                output_folder,
                                files_per_fold,
                                num_threads,
                                'fold_%d' % fold)

        # Free resources, if any
        self.settings.finalize()

    def _store_options(self, output, train_shards, val_shards, threads):
        """ Stores serialization options in a 'metadata.dat' file
        in the output directory """
        # Basic info for all datasets
        options = {
            'shards_train': train_shards,
            'shards_validation': val_shards,
            'num_threads': threads,
            'columns': self.settings.define_columns()
        }
        # We can add dataset-specific data
        extra_options = self.settings.get_options()
        if extra_options is not None:
            options.update(extra_options)
        # Save into pickle file
        save_pickle(os.path.join(output, 'metadata.dat'), options)

    def _store_dataset(self, indices_set, folder, num_shards,
                       num_threads, tag):
        """ Stores the subset selected by the row identifiers
        indices from the dataset using several threads.
         Args:
            indices_set: Set of row instances to store.
            folder: Output folder.
            num_shards: Number of slices dataset will be split.
            num_threads: Threads to use.
            tag: Dataset tag name.
        """
        # Break indices into groups such as [ranges[i][0], ranges[i][1]]
        spacing = np.linspace(0,
                              len(indices_set),
                              num_threads + 1).astype(np.int)
        # Each interval is performed by an independent thread
        ranges = [[spacing[i], spacing[i + 1]]
                  for i in range(len(spacing) - 1)]

        # Number of data slices must be multiple of the number
        # of threads for easiness
        assert not num_shards % num_threads

        logger.info('Launching %d threads for spacings: %s'
                    % (num_threads, ranges))

        # Coordinator monitors threads
        coord = tf.train.Coordinator()

        # Create threads
        threads = []
        for thread_index in range(len(ranges)):
            args = (coord, folder, tag, indices_set, thread_index,
                    ranges, num_shards, num_threads)
            threads.append(threading.Thread(target=self._process_batch,
                                            args=args))

        # Start threads
        for t in threads:
            t.start()

        # Wait for threads to end
        coord.join(threads)

        logger.info('Finished writing dataset %s (%d)'
                    % (tag, len(indices_set)))

    def _process_batch(self, coord, output_folder, name_tag, indices,
                       thread_index, ranges, num_shards, num_threads):
        """ The given thread saves batches of instances that correspond to its
        assigned range of rows in the dataset.
        Args:
            coord: Thread coordinator
            output_folder: Folder where to output the current dataset.
            name_tag: Tag of the dataset for visualization.
            indices: Total set of dataset indices to store
                (shared by all threads).
            thread_index: Thread identifier for this batch.
            ranges: List of pairs of integers specifying ranges
                instances to be processed in parallel (shared by all threads).
            num_shards: Number of slices the dataset will be split among all
                the threads.
            num_threads: Total number of threads.
        """

        def add_example(writer, example, shard_counter, counter):
            """ Dumps example into the TFRecord file """
            writer.write(example.SerializeToString())
            # increase counter
            shard_counter += 1
            counter += 1
            return shard_counter, counter

        # Each thread produces N shards, N = int(num_shards / num_threads).
        # For instance, if num_shards = 128, and the num_threads = 2,
        # then the first thread would produce shards [0, 64).
        num_shards_per_thread = int(num_shards / num_threads)

        # Instances involving each shard
        shard_ranges = np.linspace(ranges[thread_index][0],
                                   ranges[thread_index][1],
                                   num_shards_per_thread + 1).astype(int)

        # Iterate through shards of current thread
        counter, s = 0, 0
        while s < num_shards_per_thread and not coord.should_stop():

            # Generate a sharded version of the file name,
            # e.g. 'train-00002-of-00010'
            shard = thread_index * num_shards_per_thread + s
            output_filename = do.get_filename(name_tag, shard + 1, num_shards)
            output_file = os.path.join(output_folder, output_filename)

            # Create writer file
            writer = tf.python_io.TFRecordWriter(output_file)

            logger.debug('%s [thread %d]: Creating file %s \n' %
                         (datetime.now(), thread_index, output_file))

            # Select num of instances to be in the current shard file
            shard_counter = 0
            insts_per_shard = np.arange(shard_ranges[s],
                                        shard_ranges[s + 1],
                                        dtype=int)

            # Each shard iterates over a subset of the instances
            for i in insts_per_shard:

                # Get set of examples for current index and store them
                index = indices[i]
                examples = self.settings.build_examples(index)
                for ex in examples:
                    shard_counter, counter = \
                        add_example(writer, ex, shard_counter, counter)

            # Close file writer for current shard
            writer.close()

            # Increase shard counter
            s += 1

            logger.debug(
                '%s [thread %d]: Wrote %d instances to %s' %
                (datetime.now(), thread_index, shard_counter, output_file))

        # Summarize batched output
        logger.debug(
            '%s [thread %d]: Wrote %d instances to %d shards.' %
            (datetime.now(), thread_index, counter, num_shards_per_thread))


class SerializeSettings(object):

    __metaclass__ = ABCMeta

    """ Class that provides dataset-specific helpers for serialization """

    def __init__(self, data_path):
        """ Args:
                data_path: Path where to read data from
        """
        self.data_path = data_path
        self.coder = None

    def initialize(self):
        """ Initializes the settings. Resources opened here must be
        closed in finalize """
        self.coder = ImageCoder()
        self.read()

    def process_image(self, img):
        """
        Args:
            img: Image path
        Returns:
            decoded_image: Decoded image content
            height: height of the image
            width: Width of the image
            """
        return process_image(img, self.coder)

    def process_image_bytes(self, img):
        """
        Args:
            img: Ndarray of the image content
        Returns:
            Encoded image
            """
        # Rough but working solution: map matrix into temporary file
        fd, file = tempfile.mkstemp(suffix='.jpeg')
        scipy.misc.imsave(file, img)
        decoded, _, _ = process_image(file, self.coder)
        # Clean temporary file
        os.close(fd)
        os.remove(file)
        return decoded

    def image_from_url(self, url, jpeg=True):
        """ Loads an image into Tensorflow given a valid url
        Args:
            url: Link to the image (must be valid or will raise an error)
            jpeg: Whether to download image as JPEG (True) or PNG (False)
        Returns:
            decoded_image: Decoded content of the image in Tensorflow
            height: Vertical size
            width: Horizontal size
        """
        # Download image into temporary file
        suffix = '.jpeg' if jpeg else '.png'
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        download_file(url, tmp_path)

        # Add image plus additional image information
        decoded_image, height, width = self.process_image(tmp_path)

        # Free temporary resources
        os.close(fd)
        os.remove(tmp_path)

        return decoded_image, height, width

    @abc.abstractmethod
    def read(self):
        """ Reads the dataset so it is ready for being serialized """


    @abc.abstractmethod
    def get_validation_indices(self, train_ratio, val_ratio):
        """ Returns the data indices corresponding to training,
        validation and testing
        Returns
            train, val, test: training and validation index sets
        """

    @abc.abstractmethod
    def get_instance_num(self):
        """ Returns the number of instances available in the dataset """

    @abc.abstractmethod
    def define_columns(self):
        """ Returns a list of mapped columns to be stored/read into
        ExampleColumn subclasses """

    @abc.abstractmethod
    def get_options(self):
        """ Returns a dictionary of additional serialization options
        to be stored. Set to None for no extra settings """

    @abc.abstractmethod
    def build_examples(self, index):
        """ Builds TFExamples from the instance with the given index
        in the dataset.
        Returns
            examples: List of examples built from the given row index
        """

    def finalize(self):
        self.coder.finalize()
