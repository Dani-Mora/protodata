"""

Airbnb representing public  gathered from Inside Airbnb platform.
 For further information check 'airbnb_build_data.py' and 'airbnb_read_data.py'

"""

from dataio.data_ops import is_numeric, SparseColumn, ImageColumn, is_bool, \
    map_feature_type, normalize_data, bytes_feature, get_column_info, \
    is_categorical, split_data, NumericColumn, map_feature, int64_feature
from dataio.serialization_ops import SerializeSettings
from dataio.reading_ops import DataSettings
from dataio.utils import load_pickle, get_logger

import pandas as pd
import numpy as np
import tensorflow as tf
import os

logger = get_logger(__name__)


METADATA_COLUMNS = ['listing_url', 'last_scraped', 'id',
                    'scrape_id', 'picture_url']
PRICE_COLUMN = 'final_price'
AVAILABILITY_COLUMN = 'availability_365'

CSV_NAME = 'final_data.csv'
AMENITIES_NAME = 'amenities.dat'


class AirbnbSerialize(SerializeSettings):

    def __init__(self, data_folder, nq=5, subset=None, predict_price=True):
        """
        Args:
            data_path: Path where dataset is stored
            nq: Number of quantiles for numeric data normalization
            subset: Number of instances to use for the data.
                Use None to include all
            predict_price: Whether to use price as the prediction column
                (True) or the availability (False). Availability will be
                normalized into interval [0,1].
        """
        data_path = get_data_path(data_folder)
        amenities_path = get_amenities_path(data_folder)

        super(AirbnbSerialize, self).__init__(data_path)

        self.amenities_list = list(load_pickle(amenities_path)['amenities'])
        self.nq = nq
        self.num_instances = None
        self.edges, self.categories = {}, {}
        self.data = None
        self.subset = subset
        self.target = PRICE_COLUMN if predict_price else AVAILABILITY_COLUMN

        # This is a list of hardcoded bucket sizes for sparse columns.
        # If name not found here, the set of keys is extracted from the data
        self.hardcoded_buckets = {}

    def read(self):
        # Read data with pandas
        self.data = pd.read_csv(self.data_path)
        self.num_instances = self.data.shape[0]

        # Sort columns by name so the order is always the same
        self.data = self.data.reindex_axis(sorted(self.data.columns),
                                           axis=1)

        # Normalize availability into 0 and 1 when it is a target
        if self.target == AVAILABILITY_COLUMN:
            self.data[AVAILABILITY_COLUMN] = \
                self.data[AVAILABILITY_COLUMN]/365.0

        # Track set of categories for each categorical column
        for (name, dtype) in get_column_info(self.data,
                                             excluded=METADATA_COLUMNS):
            if is_categorical(dtype):
                logger.info("Tracking values in categorical column %s" % name)
                self.categories[name] = self.data[name].unique().tolist()

        # Define hardcoded buckets
        max_data = self.data.shape[0]
        self.hardcoded_buckets = {'last_scraped': max_data,
                                  'picture_url': max_data,
                                  'listing_url': max_data}

    def get_validation_indices(self, train_ratio, val_ratio):
        # Select subset of data if requested
        num = self.num_instances if self.subset is None else self.subset
        train, val, test = split_data(num, train_ratio, val_ratio)
        # Determine indices for training, validation and testing
        train_ind = self.data.index.values[train]
        val_ind = self.data.index.values[val]
        test_ind = self.data.index.values[test]
        # Normalize using training data (training + validation)
        self.data, self.edges = normalize_data(
            self.data,
            train_ind=np.concatenate([train_ind, val_ind]),
            zscores=self.nq is None,
            nq=self.nq,
            excluded=METADATA_COLUMNS + [self.target])

        return train_ind, val_ind, test_ind

    def get_options(self):
        options = {
            'data_path': self.data_path,
            'norm_metadata': self.edges,
            'amenities': self.amenities_list,
            'categories': self.categories,
            'target': self.target,
            'format': 'JPEG'
        }
        return options

    def build_examples(self, index):
        row, amenities, url = self.get_row(index=index)
        try:
            # If more than one instance wanted to be generated
            # for a single row, here is where we add them
            return [self._build_example(row, amenities, url)]
        except Exception as e:
            logger.error('Could not build instance from index {}: {}'
                         .format(index, str(e)))
            return []

    def define_columns(self):
        """ See base class """
        columns, excluded_cols = [], self.amenities_list + ['picture_url']
        col_info = get_column_info(self.data,
                                   excluded=excluded_cols)
        for (name, ftype) in col_info:

            mapped_type = map_feature_type(self.data[name].dtype)

            if is_categorical(ftype):
                # Categorical data
                if name in self.hardcoded_buckets:
                    # If hardcoded, take value
                    keys = self.hardcoded_buckets[name]
                else:
                    # Otherwise is safe to take it from the dataset itself
                    keys = self.categories[name]

                logger.info('Creating column for feature "{}" with keys "{}"'
                            .format(name, keys))
                columns.append(SparseColumn(name,
                                            keys=keys,
                                            type=mapped_type))

            elif is_numeric(ftype) or is_bool(ftype):
                logger.info('Creating column for numerical feature "{}"'
                            .format(name))
                columns.append(NumericColumn(name, type=mapped_type))

            else:
                raise RuntimeError('Unknown column type "{}" for column "{}"'
                                   .format(name, ftype))

        # Add amenities column
        am_keys = self.hardcoded_buckets['amenities'] \
            if 'amenities' in self.hardcoded_buckets \
            else self.amenities_list

        logger.info('Creating column for "amenities" with keys {}'
                    .format(am_keys))

        am_type = map_feature_type(np.dtype('object'))
        columns.append(SparseColumn('amenities',
                                    keys=am_keys,
                                    type=am_type))

        # Image-specific columns
        columns += [
            ImageColumn('image', format='JPEG'),
            NumericColumn('height', type=map_feature_type(np.dtype('int'))),
            NumericColumn('width', type=map_feature_type(np.dtype('int'))),
            SparseColumn('path', map_feature_type(np.dtype('object'))),
            SparseColumn('format', map_feature_type(np.dtype('object'))),
            SparseColumn('colorspace', map_feature_type(np.dtype('object')))
        ]

        for c in columns:
            logger.info('Creating column for feature "{}"'.format(c.name))

        return columns

    def _build_example(self, row, amenities, url):
        """ Builds an example by building a map of Proto features.
        Args:
            row: Dictionary with row information and type
            amenities: List of amenities of the example
            url: URL of the image to include
        """
        # Map features into corresponding types
        feature_dict = {name: map_feature(row[name]['value'],
                                          row[name]['type'])
                        for name, info in row.items()}

        # Create Sparse Column for the amenities of the row
        feature_dict.update({'amenities': bytes_feature(amenities)})

        # Load image from url
        decoded_image, height, width = self.image_from_url(url, jpeg=True)

        # Add image metadata
        feature_dict.update({
            'image': bytes_feature(decoded_image),
            'height': int64_feature(height),
            'width': int64_feature(width),
            'path': bytes_feature(url),
            'format': bytes_feature('JPEG'),
            'colorspace': bytes_feature('RGB')
        })

        # Build example
        return tf.train.Example(features=tf.train.Features(feature=feature_dict)) # noqa

    def get_row(self, index):
        """ Returns row information given the room identifier """
        # Get instance and copy into dict
        ref_row = self.data.ix[index].to_dict()

        # Get base features
        excluded_cols = ['picture_url'] + self.amenities_list
        feature_info = get_column_info(self.data,
                                       excluded=excluded_cols)
        row = {name: {'value': ref_row[name], 'type': ftype}
               for (name, ftype) in feature_info}

        # Get image path
        img_url = ref_row['picture_url']

        # Get amenities
        amenities = [am for am in self.amenities_list if ref_row[am]]
        return row, amenities, img_url


class AirbnbSettings(DataSettings):

    def __init__(self, dataset_location, image_specs=None,
                 embedding_dimensions=32, quantizer=None):
        """ Airbnb read settings.
        Args:
            bins: Edges to use for quantizing the price column.
                More in data_ops.py.
        """
        super(AirbnbSettings, self).__init__(
            dataset_location=dataset_location,
            image_specs=image_specs,
            embedding_dimensions=embedding_dimensions,
            quantizer=quantizer)

    def size_per_instance(self):
        return 1

    def target_class(self):
        return self.serialize_options['target']

    def _target_type(self):
        return tf.float32

    def _get_num_classes(self):
        """ Since they are real-valued targets, there is a single output """
        return 1

    def _metadata_columns(self):
        """ Returns the set of metadata columns to be excluded for training """
        return METADATA_COLUMNS + ['path', 'height', 'width',
                                   'format', 'colorspace']

    def select_wide_cols(self):
        columns = []
        for k, v in self.columns.items():

            if k not in self._metadata_columns():

                if isinstance(v, SparseColumn) or \
                        isinstance(v, NumericColumn):
                    columns.append(v.to_column())

        return columns

    def select_deep_cols(self):
        columns = []
        for k, v in self.columns.items():

            if k not in self._metadata_columns():

                if isinstance(v, SparseColumn):
                    columns.append(v.to_embedding(dims=self.embedding_dims))

                elif isinstance(v, NumericColumn):
                    columns.append(v.to_column())

        return columns


def get_data_path(folder):
    return os.path.join(folder, CSV_NAME)


def get_amenities_path(folder):
    return os.path.join(folder, AMENITIES_NAME)
