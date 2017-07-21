# tf_dataio
Repository containing wrapper functions for generating and reading from Tensorflow Proto Examples.

This code is intented to be executed using Python 3 and it is not guaranteed to work on 2.x versions. In order to use it the packages en the requirements file must be installed. We strongly recommend using virtualenv.

## Datasets

We prepared 4 datasets to be converted into the standard serialization format of Tensorflow: [diabetes](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html) (UCI, classification), [boston](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html) (UCI, regression), [MNIST](http://yann.lecun.com/exdb/mnist/) and Airbnb. 

The latter contains information related to Airbnb listings around the world. It contains metadata related to the location of the listing, the number of rooms available, services offered, etc. We consider two potentially interesting target values: availability and price (both regression problems). Before being able to use it, it must be pre-processed using the 3 notebooks included in the repository, which scraps publicly available data from the [Inside Aribnb website](http://insideairbnb.com/). 

## Serialization

In order to generate Proto Examples from any of the datasets, we must execute any of the files in the *examples* folder following this format:

> *dataset-name*_build_data.py

This will generate data into a *.tf_data* folder inside the home directory.

## Reading

In order to read Proto Examples for any of the datasets, we may also find examples within the **examples** folder. They follow this pattern:

> *dataset-name*_read_data.py

# Custom datasets

Generating Proto Examples which can be used in the defined pipeline is easy:

- For serialization, one must extend the [SerializeSettings class](https://github.com/DaniUPC/tf_dataio/blob/master/serialization_ops.py) abstract methods define in the code. The serializers for the given datasets can be found [here](https://github.com/DaniUPC/tf_dataio/tree/master/datasets).
- In order to read from them, the abstract methods from the [DataSettings class](https://github.com/DaniUPC/tf_dataio/blob/master/reading_ops.py)  must be implemented. Again, examples of this can be found [here](https://github.com/DaniUPC/tf_dataio/tree/master/datasets).
