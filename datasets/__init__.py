from protodata.datasets.airbnb import AirbnbSettings, AirbnbSerialize
from protodata.datasets.mnist import MnistSettings, MnistSerialize
from protodata.datasets.australian import AusSettings, AusSerialize
from protodata.datasets.sonar import SonarSettings, SonarSerialize
from protodata.datasets.magic import MagicSettings, MagicSerialize
from protodata.datasets.scikit_dataset import BostonSettings, \
    BostonSerialize, DiabetesSettings, DiabetesSerialize
from protodata.datasets.titanic import TitanicSettings, TitanicSerialize
from protodata.datasets.monk2 import Monk2Settings, Monk2Serialize
from protodata.datasets.balance import BalanceSettings, BalanceSerialize
from protodata.datasets.covertype import CoverTypeSettings, CoverTypeSerialize
from protodata.datasets.susy import SusySettings, SusySerialize
from protodata.datasets.quantum import QuantumSettings, QuantumSerialize
from protodata.datasets.motor import MotorSettings, MotorSerialize
from protodata.datasets.fashion import FashionMnistSettings, FashionMnistSerialize  # noqa
from protodata.datasets.cifar10 import Cifar10Settings, Cifar10Serialize


__all__ = [
    'AirbnbSettings', 'AirbnbSerialize', 'MnistSettings', 'MnistSerialize',
    'BostonSettings', 'BostonSerialize', 'AusSettings', 'AusSerialize',
    'SonarSettings', 'SonarSerialize', 'DiabetesSettings', 'DiabetesSerialize',  # noqa
    'MagicSettings', 'MagicSerialize', 'TitanicSettings', 'TitanicSerialize',
    'Monk2Settings', 'Monk2Serialize', 'MotorSettings', 'MotorSerialize',
    'BalanceSettings', 'BalanceSerialize', 'CoverTypeSettings',
    'CoverTypeSerialize', 'SusySerialize', 'SusySettings', 'QuantumSerialize',
    'QuantumSettings', 'FashionMnistSettings', 'FashionMnistSerialize',
    'Cifar10Serialize', 'Cifar10Settings', 'Datasets'
]


class Datasets:
    AUS = 'australian'
    AIRBNB_PRICE = 'airbnb_price'
    AIRBNB_AVAILABLE = 'airbnb_availability'
    BALANCE = 'balance'
    BOSTON = 'boston'
    CIFAR10 = 'cifar10'
    COVERTYPE = 'covertype'
    DIABETES = 'diabetes'
    MAGIC = 'magic'
    FASHION_MNIST = 'fashion'
    MNIST = 'mnist'
    MONK2 = 'monk2'
    MOTOR = 'motor'
    QUANTUM = 'quantum'
    SONAR = 'sonar'
    SUSY = 'susy'
    TITANIC = 'titanic'
