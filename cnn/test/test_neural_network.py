import os
import random
import sys
import unittest

cnn_package_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(cnn_package_path)
from cnn import neural_network, convolution_layer, subsampling_layer, output_layer
from cnn.common import integer_to_ordinal


class TestNeuralNetwork(unittest.TestCase):
    def test_cnn_creation_positive_1(self):
        pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
