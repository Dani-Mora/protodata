"""

Serialization of the Airbnb dataset into Example protos.

 The original data is converted into subdirectories that contain a
 moderate amount of files for both training and evaluation. Each file
 contains several Example Protos.

  Some of the columns contained in the examples:

    - accommodates: Number of rooms offered by the host.
    - area: City or city area where the listing comes from.
    - bed_type: Type of bed.
    - bedrooms: Number of bedrooms of the offer.
    - cancelation_policy: Type of cancellation policy.
    - country: Country the listing belongs to.
    - final_price: Final price per night of the listing.
    - minimum_nights: Minimum number of nights.
    - property_type: Type of property.
    - recent_review: Whether the listing had any review in the last 30 days.
    - reviews_per_month: Monthly average of reviews.
    - review_scores_accuracy: Accuracy mean review score.
        Categories: non-rated, bad, normal, good.
    - review_scores_checkin: Check-in mean review score.
        Categories: non-rated, bad, normal, good.
    - review_scores_cleanliness: Cleanliness mean review score.
        Categories: non-rated, bad, normal, good.
    - room_type: Type of room offered.
    - sec_deposit: Amount of the security deposit.
    - state: State the listing belongs to.
    - subarea: Area/Neighbourhood the listing is located in.
    - amenities: Set of amenities/services that are offered by the
        host (e.g. wi-fi, dryer)
    - image: Cover image of the listing.

Some additonal metadata columns have been included but are not used for
training, such as the identifier of the listing, the time it was scraped
or some metadata of the image (e.g. dimensions, url, format).

For code simplicity and visualization, generation of this dataset has been
ported into 3 notebooks (folder called 'notebooks'). Execute them to generate
the needed raw data for this dataset. It takes some time to download it, so
be patient :)

"""

from protodata.serialization_ops import DataSerializer
from protodata.datasets import AirbnbSerialize, Datasets
from protodata.utils import get_tmp_data_location, get_data_location

import tensorflow as tf
import os

# Note AIRBNB.AVAILABLE can be also generated in the notebooks
DATASET = Datasets.AIRBNB_PRICE

# Data paths
tf.app.flags.DEFINE_string('raw_data_location',
                           get_tmp_data_location(DATASET),
                           'Where raw data is located')

tf.app.flags.DEFINE_string('data_location',
                           get_data_location(DATASET),
                           'Where to store data')

# Data parameters
tf.app.flags.DEFINE_integer('nq',
                            None,
                            'Number of quantiles for numeric normalization.'
                            'Set to None to use zscores')

tf.app.flags.DEFINE_integer('subset',
                            3000,
                            'Number of instances to use. Set to None for all')

tf.app.flags.DEFINE_float('train_ratio', 0.80, 'Ratio of training instances')
tf.app.flags.DEFINE_float('val_ratio', 0.10, 'Ratio of validation instances')

# Serialization parameters
tf.app.flags.DEFINE_integer('train_shards',
                            64,
                            'Number of files in training TFRecord files.')

tf.app.flags.DEFINE_integer('validation_shards',
                            8,
                            'Number of files in validation TFRecord files.')

tf.app.flags.DEFINE_integer('test_shards',
                            8,
                            'Number of files in testing TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads',
                            4,
                            'Number of threads to preprocess the images.')

FLAGS = tf.app.flags.FLAGS


if __name__ == '__main__':

    if not os.path.isdir(FLAGS.raw_data_location):
        raise RuntimeError('Airbnb data has not been generated.' +
                           'Please go to the notebooks folder and execute ' +
                           'the three available notebooks before continuing')

    # Configuration for extraction
    use_price = DATASET == Datasets.AIRBNB_PRICE
    settings = AirbnbSerialize(data_folder=FLAGS.raw_data_location,
                               nq=FLAGS.nq,
                               subset=FLAGS.subset,
                               predict_price=use_price)

    # Save to TFRecord
    serializer = DataSerializer(settings)

    # Serialize data
    serializer.serialize(output_folder=FLAGS.data_location,
                         train_ratio=FLAGS.train_ratio,
                         val_ratio=FLAGS.val_ratio,
                         num_threads=FLAGS.num_threads,
                         train_shards=FLAGS.train_shards,
                         val_shards=FLAGS.validation_shards,
                         test_shards=FLAGS.test_shards)
