from dataio.datasets.airbnb import AirbnbSettings, AirbnbSerialize
from dataio.datasets.mnist import MnistSettings, MnistSerialize
from dataio.datasets.scikit_dataset import BostonSettings, BostonSerialize, \
    DiabetesSettings, DiabetesSerialize

__all__ = ['AirbnbSettings', 'AirbnbSerialize', 'MnistSettings',
           'MnistSerialize', 'BostonSettings', 'BostonSerialize',
           'DiabetesSettings', 'DiabetesSerialize', 'Datasets']


class Datasets:
    AIRBNB_PRICE = 'airbnb_price'
    AIRBNB_AVAILABLE = 'airbnb_availability'
    MNIST = 'mnist'
    BOSTON = 'boston'
    DIABETES = 'diabetes'
