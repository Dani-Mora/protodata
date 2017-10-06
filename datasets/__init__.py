from protodata.datasets.airbnb import AirbnbSettings, AirbnbSerialize
from protodata.datasets.mnist import MnistSettings, MnistSerialize
from protodata.datasets.australian import AusSettings, AusSerialize
from protodata.datasets.sonar import SonarSettings, SonarSerialize
from protodata.datasets.magic import MagicSettings, MagicSerialize
from protodata.datasets.scikit_dataset import BostonSettings, \
    BostonSerialize, DiabetesSettings, DiabetesSerialize
from protodata.datasets.titanic import TitanicSettings, TitanicSerialize


__all__ = ['AirbnbSettings', 'AirbnbSerialize', 'MnistSettings',
           'MnistSerialize', 'BostonSettings', 'BostonSerialize',
           'AusSettings', 'AusSerialize', 'SonarSettings', 'SonarSerialize',
           'DiabetesSettings', 'DiabetesSerialize', 'MagicSettings',
           'MagicSerialize', 'TitanicSettings', 'TitanicSerialize',
           'Datasets']


class Datasets:
    AUS = 'australian'
    AIRBNB_PRICE = 'airbnb_price'
    AIRBNB_AVAILABLE = 'airbnb_availability'
    MNIST = 'mnist'
    BOSTON = 'boston'
    DIABETES = 'diabetes'
    TITANIC = 'titanic'
    MAGIC = 'magic'
    SONAR = 'sonar'
