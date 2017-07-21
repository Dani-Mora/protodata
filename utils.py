# ! /usr/bin/python

import pickle
import os
import json
import urllib.request
import logging
import sys
from datetime import datetime


def get_logger(name, level=logging.DEBUG, channel=sys.stdout):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Set handler level and set output to go into std output
    handler = logging.StreamHandler(channel)
    handler.setLevel(level)

    # Set specific format
    src_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(src_format)
    handler.setFormatter(formatter)

    # Add handler to main logger
    logger.addHandler(handler)
    return logger


logger = get_logger(__name__)


NON_SAVABLE_COL = 'not_saveable'
HOME_FOLDER = os.path.expanduser("~")
DATA_ROOT = os.path.join(HOME_FOLDER, '.tf_data')
TMP_ROOT = os.path.join(HOME_FOLDER, '.tf_data_tmp')


class NetworkModels:
    ALEXNET = 'alexnet'
    VGG = 'vgg'
    MNIST = 'mnist'  # Simple model to train mnist


def get_data_root():
    return DATA_ROOT


def get_tmp_root():
    return TMP_ROOT


def get_data_location(dataset):
    return os.path.join(get_data_root(), dataset)


def get_tmp_data_location(dataset):
    return os.path.join(get_tmp_root(), dataset)


def download_file(file_url, path):
    """ Downloads a file into the respective path """
    urllib.request.urlretrieve(file_url, path)


def load_pickle(path):
    """ Loads object from a pickle file """
    with open(path, 'rb') as f:
        content = pickle.load(f)
    return content


def save_pickle(path, obj):
    """ Loads object from a pickle file """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def opt_save_pickle(path, obj):
    """ Saves the object as pickle if path is not None """
    if path is not None:
        save_pickle(path, obj)


def read_json(path):
    """ Reads metadata from JSON file and dumps it into memory.
    If file is too big, consider using json for json streaming. """
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def save_json(path, obj):
    with open(path, 'wb') as f:
        json.dump(obj, f)


def opt_save_json(path, obj):
    """ Saves the object as pickle if path is not None """
    if path is not None:
        save_json(path, obj)


def create_dir(path):
    """ Creates directory if it does not exist """
    if not os.path.exists(path):
        os.makedirs(path)


def download(url, dst):
    """ Downloads the content in the given url.
    If path provided, downloads into that path
    """
    urllib.request.urlretrieve(url, dst)


def get_subfolders(folder):
    """ Returns the folders contained in the input path """
    return [name for name in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, name))]


def to_date(x):
    return datetime.strptime(x, '%Y-%m-%d')


def get_filename_url(url):
    """
    Args:
        url: Input url
    Returns:
        Returns the corresponding file name and its extension
    """
    name = os.path.basename(url)
    return os.path.splitext(name)


class FileNotFound(AttributeError):
    pass
