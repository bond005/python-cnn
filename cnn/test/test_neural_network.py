import os
import re
import sys
import unittest

import numpy

cnn_package_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(cnn_package_path)
from cnn import neural_network, convolution_layer, subsampling_layer, output_layer
from cnn.neural_network import ECNNCreating, ECNNTraining


class TestNeuralNetwork(unittest.TestCase):
    def create_train_set(self, number_of_input_maps=1, size_of_input_map=(28, 28)):
        if number_of_input_maps > 0:
            X_train = numpy.empty(shape=(100, number_of_input_maps, size_of_input_map[0], size_of_input_map[1]),
                                  dtype=numpy.float32)
        else:
            X_train = numpy.empty(shape=(100, size_of_input_map[0], size_of_input_map[1]), dtype=numpy.float32)
        y_train = numpy.empty(shape=(100,), dtype=numpy.uint32)
        for class_ind in range(10):
            rand_mean = numpy.random.uniform(-0.5, 0.5)
            rand_std = numpy.random.uniform(0.1, 0.5)
            if number_of_input_maps > 0:
                for input_map_ind in range(number_of_input_maps):
                    random_samples = numpy.random.normal(rand_mean, rand_std,
                                                         size=(10, size_of_input_map[0], size_of_input_map[1]))
                    for sample_ind in range(10):
                        X_train[class_ind * 10 + sample_ind][input_map_ind] = random_samples[sample_ind]
            else:
                random_samples = numpy.random.normal(rand_mean, rand_std,
                                                     size=(10, size_of_input_map[0], size_of_input_map[1]))
                for sample_ind in range(10):
                    X_train[class_ind * 10 + sample_ind] = random_samples[sample_ind]
            for sample_ind in range(10):
                y_train[class_ind * 10 + sample_ind] = class_ind
        return X_train, y_train

    def create_cnn(self, number_of_input_maps=1):
        return neural_network.CNNClassifier(
            input_maps_number=number_of_input_maps, input_map_size=(28, 28),
            structure_of_hidden_layers=(
                {
                    'feature_maps_number': 4,
                    'feature_map_size': (24, 24),
                    'receptive_field_size': (5, 5)
                },
                {
                    'feature_maps_number': 4,
                    'feature_map_size': (12, 12),
                    'receptive_field_size': (2, 2)
                },
                {
                    'feature_maps_number': 12,
                    'feature_map_size': (8, 8),
                    'receptive_field_size': (5, 5)
                },
                {
                    'feature_maps_number': 12,
                    'feature_map_size': (4, 4),
                    'receptive_field_size': (2, 2)
                }
            ),
            classes_number=10,
            max_epochs_number=5,
            learning_rate=0.01,
            early_stopping=False,
            validation_fraction=0.15
        )

    def test_cnn_creation_positive_01(self):
        """ Проверяем, что СНС нормально создаётся. """
        cnn = self.create_cnn()
        self.assertEqual(cnn.input_maps_number, 1)
        self.assertIsInstance(cnn.input_map_size, tuple)
        self.assertEqual(cnn.input_map_size, (28, 28))
        self.assertEqual(cnn.max_train_epochs, 5)
        self.assertIsInstance(cnn.layers, list)
        self.assertEqual(len(cnn.layers), 5)
        self.assertIsInstance(cnn.layers[0], convolution_layer.ConvolutionLayer)
        self.assertIsInstance(cnn.layers[1], subsampling_layer.SubsamplingLayer)
        self.assertIsInstance(cnn.layers[2], convolution_layer.ConvolutionLayer)
        self.assertIsInstance(cnn.layers[3], subsampling_layer.SubsamplingLayer)
        self.assertIsInstance(cnn.layers[4], output_layer.OutputLayer)
        self.assertEqual(cnn.layers[4].neurons_number, 10)
        self.assertIsInstance(cnn.learning_rate, float)
        self.assertAlmostEqual(cnn.learning_rate, 0.01, places=7)
        self.assertIsInstance(cnn.early_stopping, bool)
        self.assertFalse(cnn.early_stopping)
        self.assertIsInstance(cnn.validation_fraction, float)
        self.assertAlmostEqual(cnn.validation_fraction, 0.15, places=7)

    def test_cnn_creation_negative_01(self):
        """ Проверяем генерацию исключения при некорректно указанном количестве эпох обучения. """
        target_err_msg = 'Maximal number of training epochs is specified incorrectly!'
        with self.assertRaisesRegex(ECNNCreating, re.escape(target_err_msg)):
            neural_network.CNNClassifier(
                input_maps_number=1, input_map_size=(28, 28),
                structure_of_hidden_layers=(
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (24, 24),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (12, 12),
                        'receptive_field_size': (2, 2)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (8, 8),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (4, 4),
                        'receptive_field_size': (2, 2)
                    }
                ),
                classes_number=10,
                max_epochs_number=-50,
                learning_rate=0.01,
                early_stopping=False,
                validation_fraction=0.15
            )

    def test_cnn_creation_negative_02(self):
        """ Проверяем генерацию исключения при некорректно указанном коэффициенте скорости обучения. """
        target_err_msg = 'Learning rate parameter is specified incorrectly!'
        with self.assertRaisesRegex(ECNNCreating, re.escape(target_err_msg)):
            neural_network.CNNClassifier(
                input_maps_number=1, input_map_size=(28, 28),
                structure_of_hidden_layers=(
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (24, 24),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (12, 12),
                        'receptive_field_size': (2, 2)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (8, 8),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (4, 4),
                        'receptive_field_size': (2, 2)
                    }
                ),
                classes_number=10,
                max_epochs_number=50,
                learning_rate=0.0,
                early_stopping=False,
                validation_fraction=0.15
            )

    def test_cnn_creation_negative_03(self):
        """ Проверяем генерацию исключения при слишком малой доле обучающего множества для early stopping. """
        target_err_msg = 'Validation fraction is specified incorrectly!'
        with self.assertRaisesRegex(ECNNCreating, re.escape(target_err_msg)):
            neural_network.CNNClassifier(
                input_maps_number=1, input_map_size=(28, 28),
                structure_of_hidden_layers=(
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (24, 24),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (12, 12),
                        'receptive_field_size': (2, 2)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (8, 8),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (4, 4),
                        'receptive_field_size': (2, 2)
                    }
                ),
                classes_number=10,
                max_epochs_number=50,
                learning_rate=0.01,
                early_stopping=False,
                validation_fraction=0.0
            )

    def test_cnn_creation_negative_04(self):
        """ Проверяем генерацию исключения при слишком большой доле обучающего множества для early stopping. """
        target_err_msg = 'Validation fraction is specified incorrectly!'
        with self.assertRaisesRegex(ECNNCreating, re.escape(target_err_msg)):
            neural_network.CNNClassifier(
                input_maps_number=1, input_map_size=(28, 28),
                structure_of_hidden_layers=(
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (24, 24),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (12, 12),
                        'receptive_field_size': (2, 2)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (8, 8),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (4, 4),
                        'receptive_field_size': (2, 2)
                    }
                ),
                classes_number=10,
                max_epochs_number=50,
                learning_rate=0.01,
                early_stopping=False,
                validation_fraction=1.0
            )

    def test_cnn_creation_negative_05(self):
        """ Проверяем генерацию исключения при некорректно указанном количестве входных карт. """
        target_err_msg = 'Input maps for CNN are specified incorrectly!'
        with self.assertRaisesRegex(ECNNCreating, re.escape(target_err_msg)):
            neural_network.CNNClassifier(
                input_maps_number=0, input_map_size=(28, 28),
                structure_of_hidden_layers=(
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (24, 24),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (12, 12),
                        'receptive_field_size': (2, 2)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (8, 8),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (4, 4),
                        'receptive_field_size': (2, 2)
                    }
                ),
                classes_number=10,
                max_epochs_number=50,
                learning_rate=0.01,
                early_stopping=False,
                validation_fraction=0.15
            )

    def test_cnn_creation_negative_06(self):
        """ Проверяем генерацию исключения, когда входная карта не двумерна. """
        target_err_msg = 'Input maps for CNN are specified incorrectly!'
        with self.assertRaisesRegex(ECNNCreating, re.escape(target_err_msg)):
            neural_network.CNNClassifier(
                input_maps_number=1, input_map_size=(28, 28, 28),
                structure_of_hidden_layers=(
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (24, 24),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (12, 12),
                        'receptive_field_size': (2, 2)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (8, 8),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (4, 4),
                        'receptive_field_size': (2, 2)
                    }
                ),
                classes_number=10,
                max_epochs_number=50,
                learning_rate=0.01,
                early_stopping=False,
                validation_fraction=0.15
            )

    def test_cnn_creation_negative_07(self):
        """ Проверяем генерацию исключения, когда входная карта двумерна, но одно из её измерений нулевое. """
        target_err_msg = 'Input maps for CNN are specified incorrectly!'
        with self.assertRaisesRegex(ECNNCreating, re.escape(target_err_msg)):
            neural_network.CNNClassifier(
                input_maps_number=1, input_map_size=(28, 0),
                structure_of_hidden_layers=(
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (24, 24),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (12, 12),
                        'receptive_field_size': (2, 2)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (8, 8),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (4, 4),
                        'receptive_field_size': (2, 2)
                    }
                ),
                classes_number=10,
                max_epochs_number=50,
                learning_rate=0.01,
                early_stopping=False,
                validation_fraction=0.15
            )

    def test_cnn_creation_negative_08(self):
        """ Проверяем генерацию исключения при некорректно указанном количестве распознаваемых классов. """
        target_err_msg = 'Number of classes is too small!'
        with self.assertRaisesRegex(ECNNCreating, re.escape(target_err_msg)):
            neural_network.CNNClassifier(
                input_maps_number=1, input_map_size=(28, 28),
                structure_of_hidden_layers=(
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (24, 24),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (12, 12),
                        'receptive_field_size': (2, 2)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (8, 8),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (4, 4),
                        'receptive_field_size': (2, 2)
                    }
                ),
                classes_number=1,
                max_epochs_number=50,
                learning_rate=0.01,
                early_stopping=False,
                validation_fraction=0.15
            )

    def test_cnn_creation_negative_09(self):
        """ Проверяем генерацию исключения, когда скрытых слоёв слишком мало. """
        target_err_msg = 'Number of hidden layers is too small!'
        with self.assertRaisesRegex(ECNNCreating, re.escape(target_err_msg)):
            neural_network.CNNClassifier(
                input_maps_number=1, input_map_size=(28, 28),
                structure_of_hidden_layers=(
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (24, 24),
                        'receptive_field_size': (5, 5)
                    },
                ),
                classes_number=10,
                max_epochs_number=50,
                learning_rate=0.01,
                early_stopping=False,
                validation_fraction=0.15
            )

    def test_cnn_creation_negative_10(self):
        """ Проверяем генерацию исключения, когда количество скрытых слоёв нечётно. """
        target_err_msg = 'Number of hidden layers is incorrect (it must be even value)!'
        with self.assertRaisesRegex(ECNNCreating, re.escape(target_err_msg)):
            neural_network.CNNClassifier(
                input_maps_number=1, input_map_size=(28, 28),
                structure_of_hidden_layers=(
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (24, 24),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (12, 12),
                        'receptive_field_size': (2, 2)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (8, 8),
                        'receptive_field_size': (5, 5)
                    }
                ),
                classes_number=10,
                max_epochs_number=50,
                learning_rate=0.01,
                early_stopping=False,
                validation_fraction=0.15
            )

    def test_cnn_creation_negative_11(self):
        """ Проверяем генерацию исключения, когда в описании структуры слоя забыт параметр `feature_maps_number`. """
        target_err_msg = 'Structure of layer 4 is specified incorrectly!'
        with self.assertRaisesRegex(ECNNCreating, re.escape(target_err_msg)):
            neural_network.CNNClassifier(
                input_maps_number=1, input_map_size=(28, 28),
                structure_of_hidden_layers=(
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (24, 24),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (12, 12),
                        'receptive_field_size': (2, 2)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (8, 8),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_map_size': (4, 4),
                        'receptive_field_size': (2, 2)
                    }
                ),
                classes_number=10,
                max_epochs_number=50,
                learning_rate=0.01,
                early_stopping=False,
                validation_fraction=0.15
            )

    def test_cnn_creation_negative_12(self):
        """ Проверяем генерацию исключения, когда в описании структуры слоя забыт параметр `feature_map_size`. """
        target_err_msg = 'Structure of layer 4 is specified incorrectly!'
        with self.assertRaisesRegex(ECNNCreating, re.escape(target_err_msg)):
            neural_network.CNNClassifier(
                input_maps_number=1, input_map_size=(28, 28),
                structure_of_hidden_layers=(
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (24, 24),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (12, 12),
                        'receptive_field_size': (2, 2)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (8, 8),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 12,
                        'receptive_field_size': (2, 2)
                    }
                ),
                classes_number=10,
                max_epochs_number=50,
                learning_rate=0.01,
                early_stopping=False,
                validation_fraction=0.15
            )

    def test_cnn_creation_negative_13(self):
        """ Проверяем генерацию исключения, когда в описании структуры слоя забыт параметр `receptive_field_size`. """
        target_err_msg = 'Structure of layer 4 is specified incorrectly!'
        with self.assertRaisesRegex(ECNNCreating, re.escape(target_err_msg)):
            neural_network.CNNClassifier(
                input_maps_number=1, input_map_size=(28, 28),
                structure_of_hidden_layers=(
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (24, 24),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (12, 12),
                        'receptive_field_size': (2, 2)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (8, 8),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (4, 4)
                    }
                ),
                classes_number=10,
                max_epochs_number=50,
                learning_rate=0.01,
                early_stopping=False,
                validation_fraction=0.15
            )

    def test_cnn_creation_negative_14(self):
        """ Проверяем генерацию исключения, когда структура первого слоя не соответствует структуре входных карт. """
        target_err_msg = 'Structure of layer 1 does not correspond to structure of input maps!'
        with self.assertRaisesRegex(ECNNCreating, re.escape(target_err_msg)):
            neural_network.CNNClassifier(
                input_maps_number=1, input_map_size=(28, 28),
                structure_of_hidden_layers=(
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (22, 22),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (12, 12),
                        'receptive_field_size': (2, 2)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (8, 8),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (4, 4),
                        'receptive_field_size': (2, 2)
                    }
                ),
                classes_number=10,
                max_epochs_number=50,
                learning_rate=0.01,
                early_stopping=False,
                validation_fraction=0.15
            )

    def test_cnn_creation_negative_15(self):
        """ Проверяем генерацию исключения, когда слой подвыборки не соответствует предшествующему слою свёртки. """
        target_err_msg = 'Structure of layer 2 does not correspond to structure of previous layer!'
        with self.assertRaisesRegex(ECNNCreating, re.escape(target_err_msg)):
            neural_network.CNNClassifier(
                input_maps_number=1, input_map_size=(28, 28),
                structure_of_hidden_layers=(
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (24, 24),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (12, 12),
                        'receptive_field_size': (2, 3)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (8, 8),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (4, 4),
                        'receptive_field_size': (2, 2)
                    }
                ),
                classes_number=10,
                max_epochs_number=50,
                learning_rate=0.01,
                early_stopping=False,
                validation_fraction=0.15
            )

    def test_cnn_creation_negative_16(self):
        """ Проверяем генерацию исключения, когда слой подвыборки не соответствует предшествующему слою свёртки. """
        target_err_msg = 'Structure of layer 2 does not correspond to structure of previous layer!'
        with self.assertRaisesRegex(ECNNCreating, re.escape(target_err_msg)):
            neural_network.CNNClassifier(
                input_maps_number=1, input_map_size=(28, 28),
                structure_of_hidden_layers=(
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (24, 24),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 3,
                        'feature_map_size': (12, 12),
                        'receptive_field_size': (2, 2)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (8, 8),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (4, 4),
                        'receptive_field_size': (2, 2)
                    }
                ),
                classes_number=10,
                max_epochs_number=50,
                learning_rate=0.01,
                early_stopping=False,
                validation_fraction=0.15
            )

    def test_cnn_creation_negative_17(self):
        """ Проверяем генерацию исключения, когда слой свёртки не соответствует предшествующему слою подвыборки. """
        target_err_msg = 'Structure of layer 3 does not correspond to structure of previous layer!'
        with self.assertRaisesRegex(ECNNCreating, re.escape(target_err_msg)):
            neural_network.CNNClassifier(
                input_maps_number=1, input_map_size=(28, 28),
                structure_of_hidden_layers=(
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (24, 24),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (12, 12),
                        'receptive_field_size': (2, 2)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (8, 7),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (4, 4),
                        'receptive_field_size': (2, 2)
                    }
                ),
                classes_number=10,
                max_epochs_number=50,
                learning_rate=0.01,
                early_stopping=False,
                validation_fraction=0.15
            )

    def test_cnn_creation_negative_18(self):
        """ Проверяем генерацию исключения, когда слой подвыборки (последний) не соответствует слою свёртки перед ним.
        """
        target_err_msg = 'Structure of layer 4 does not correspond to structure of previous layer!'
        with self.assertRaisesRegex(ECNNCreating, re.escape(target_err_msg)):
            neural_network.CNNClassifier(
                input_maps_number=1, input_map_size=(28, 28),
                structure_of_hidden_layers=(
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (24, 24),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 4,
                        'feature_map_size': (12, 12),
                        'receptive_field_size': (2, 2)
                    },
                    {
                        'feature_maps_number': 12,
                        'feature_map_size': (8, 8),
                        'receptive_field_size': (5, 5)
                    },
                    {
                        'feature_maps_number': 13,
                        'feature_map_size': (4, 4),
                        'receptive_field_size': (2, 2)
                    }
                ),
                classes_number=10,
                max_epochs_number=50,
                learning_rate=0.01,
                early_stopping=False,
                validation_fraction=0.15
            )

    def test_max_train_epochs_positive_1(self):
        """ Проверяем работу свойства max_train_epochs при установке корректного значения. """
        cnn = neural_network.CNNClassifier(
            input_maps_number=1, input_map_size=(28, 28),
            structure_of_hidden_layers=(
                {
                    'feature_maps_number': 4,
                    'feature_map_size': (24, 24),
                    'receptive_field_size': (5, 5)
                },
                {
                    'feature_maps_number': 4,
                    'feature_map_size': (12, 12),
                    'receptive_field_size': (2, 2)
                },
                {
                    'feature_maps_number': 12,
                    'feature_map_size': (8, 8),
                    'receptive_field_size': (5, 5)
                },
                {
                    'feature_maps_number': 12,
                    'feature_map_size': (4, 4),
                    'receptive_field_size': (2, 2)
                }
            ),
            classes_number=10,
            max_epochs_number=50,
            learning_rate=0.01,
            early_stopping=False,
            validation_fraction=0.15
        )
        cnn.max_train_epochs = 100
        self.assertEqual(cnn.max_train_epochs, 100)

    def test_max_train_epochs_negative_1(self):
        """ Проверяем работу свойства max_train_epochs при установке не целого, а вещественного значения. """
        target_err_msg = 'Maximal number of training epochs is specified incorrectly!'
        cnn = neural_network.CNNClassifier(
            input_maps_number=1, input_map_size=(28, 28),
            structure_of_hidden_layers=(
                {
                    'feature_maps_number': 4,
                    'feature_map_size': (24, 24),
                    'receptive_field_size': (5, 5)
                },
                {
                    'feature_maps_number': 4,
                    'feature_map_size': (12, 12),
                    'receptive_field_size': (2, 2)
                },
                {
                    'feature_maps_number': 12,
                    'feature_map_size': (8, 8),
                    'receptive_field_size': (5, 5)
                },
                {
                    'feature_maps_number': 12,
                    'feature_map_size': (4, 4),
                    'receptive_field_size': (2, 2)
                }
            ),
            classes_number=10,
            max_epochs_number=50,
            learning_rate=0.01,
            early_stopping=False,
            validation_fraction=0.15
        )
        with self.assertRaisesRegex(TypeError, re.escape(target_err_msg)):
            cnn.max_train_epochs = 50.5

    def test_max_train_epochs_negative_2(self):
        """ Проверяем работу свойства max_train_epochs при установке целого, но не положительного значения. """
        target_err_msg = 'Maximal number of training epochs is specified incorrectly!'
        cnn = neural_network.CNNClassifier(
            input_maps_number=1, input_map_size=(28, 28),
            structure_of_hidden_layers=(
                {
                    'feature_maps_number': 4,
                    'feature_map_size': (24, 24),
                    'receptive_field_size': (5, 5)
                },
                {
                    'feature_maps_number': 4,
                    'feature_map_size': (12, 12),
                    'receptive_field_size': (2, 2)
                },
                {
                    'feature_maps_number': 12,
                    'feature_map_size': (8, 8),
                    'receptive_field_size': (5, 5)
                },
                {
                    'feature_maps_number': 12,
                    'feature_map_size': (4, 4),
                    'receptive_field_size': (2, 2)
                }
            ),
            classes_number=10,
            max_epochs_number=50,
            learning_rate=0.01,
            early_stopping=False,
            validation_fraction=0.15
        )
        with self.assertRaisesRegex(TypeError, re.escape(target_err_msg)):
            cnn.max_train_epochs = 0

    def test_early_stopping_positive_1(self):
        """ Проверяем работу свойства early_stopping при установке корректного значения. """
        cnn = neural_network.CNNClassifier(
            input_maps_number=1, input_map_size=(28, 28),
            structure_of_hidden_layers=(
                {
                    'feature_maps_number': 4,
                    'feature_map_size': (24, 24),
                    'receptive_field_size': (5, 5)
                },
                {
                    'feature_maps_number': 4,
                    'feature_map_size': (12, 12),
                    'receptive_field_size': (2, 2)
                },
                {
                    'feature_maps_number': 12,
                    'feature_map_size': (8, 8),
                    'receptive_field_size': (5, 5)
                },
                {
                    'feature_maps_number': 12,
                    'feature_map_size': (4, 4),
                    'receptive_field_size': (2, 2)
                }
            ),
            classes_number=10,
            max_epochs_number=50,
            learning_rate=0.01,
            early_stopping=True,
            validation_fraction=0.15
        )
        cnn.early_stopping = False
        self.assertFalse(cnn.early_stopping)

    def test_early_stopping_negative_1(self):
        """ Проверяем работу свойства early_stopping при установке не булевского, а целочисленного значения. """
        target_err_msg = 'Usage of early stopping is specified incorrectly!'
        cnn = neural_network.CNNClassifier(
            input_maps_number=1, input_map_size=(28, 28),
            structure_of_hidden_layers=(
                {
                    'feature_maps_number': 4,
                    'feature_map_size': (24, 24),
                    'receptive_field_size': (5, 5)
                },
                {
                    'feature_maps_number': 4,
                    'feature_map_size': (12, 12),
                    'receptive_field_size': (2, 2)
                },
                {
                    'feature_maps_number': 12,
                    'feature_map_size': (8, 8),
                    'receptive_field_size': (5, 5)
                },
                {
                    'feature_maps_number': 12,
                    'feature_map_size': (4, 4),
                    'receptive_field_size': (2, 2)
                }
            ),
            classes_number=10,
            max_epochs_number=50,
            learning_rate=0.01,
            early_stopping=True,
            validation_fraction=0.15
        )
        with self.assertRaisesRegex(TypeError, re.escape(target_err_msg)):
            cnn.early_stopping = 0

    def test_validation_fraction_positive_1(self):
        """ Проверяем работу свойства validation_fraction при установке корректного значения. """
        cnn = neural_network.CNNClassifier(
            input_maps_number=1, input_map_size=(28, 28),
            structure_of_hidden_layers=(
                {
                    'feature_maps_number': 4,
                    'feature_map_size': (24, 24),
                    'receptive_field_size': (5, 5)
                },
                {
                    'feature_maps_number': 4,
                    'feature_map_size': (12, 12),
                    'receptive_field_size': (2, 2)
                },
                {
                    'feature_maps_number': 12,
                    'feature_map_size': (8, 8),
                    'receptive_field_size': (5, 5)
                },
                {
                    'feature_maps_number': 12,
                    'feature_map_size': (4, 4),
                    'receptive_field_size': (2, 2)
                }
            ),
            classes_number=10,
            max_epochs_number=50,
            learning_rate=0.01,
            early_stopping=True,
            validation_fraction=0.15
        )
        cnn.validation_fraction = 0.2
        self.assertAlmostEqual(cnn.validation_fraction, 0.2, places=7)

    def test_validation_fraction_negative_1(self):
        """ Проверяем работу свойства validation_fraction при установке не вещественного, а строкового значения. """
        target_err_msg = 'Validation fraction is specified incorrectly!'
        cnn = neural_network.CNNClassifier(
            input_maps_number=1, input_map_size=(28, 28),
            structure_of_hidden_layers=(
                {
                    'feature_maps_number': 4,
                    'feature_map_size': (24, 24),
                    'receptive_field_size': (5, 5)
                },
                {
                    'feature_maps_number': 4,
                    'feature_map_size': (12, 12),
                    'receptive_field_size': (2, 2)
                },
                {
                    'feature_maps_number': 12,
                    'feature_map_size': (8, 8),
                    'receptive_field_size': (5, 5)
                },
                {
                    'feature_maps_number': 12,
                    'feature_map_size': (4, 4),
                    'receptive_field_size': (2, 2)
                }
            ),
            classes_number=10,
            max_epochs_number=50,
            learning_rate=0.01,
            early_stopping=True,
            validation_fraction=0.15
        )
        with self.assertRaisesRegex(TypeError, re.escape(target_err_msg)):
            cnn.validation_fraction = '0.15'

    def test_validation_fraction_negative_2(self):
        """ Проверяем работу свойства validation_fraction при установке слишком малого вещественного значения. """
        target_err_msg = 'Validation fraction is specified incorrectly!'
        cnn = neural_network.CNNClassifier(
            input_maps_number=1, input_map_size=(28, 28),
            structure_of_hidden_layers=(
                {
                    'feature_maps_number': 4,
                    'feature_map_size': (24, 24),
                    'receptive_field_size': (5, 5)
                },
                {
                    'feature_maps_number': 4,
                    'feature_map_size': (12, 12),
                    'receptive_field_size': (2, 2)
                },
                {
                    'feature_maps_number': 12,
                    'feature_map_size': (8, 8),
                    'receptive_field_size': (5, 5)
                },
                {
                    'feature_maps_number': 12,
                    'feature_map_size': (4, 4),
                    'receptive_field_size': (2, 2)
                }
            ),
            classes_number=10,
            max_epochs_number=50,
            learning_rate=0.01,
            early_stopping=True,
            validation_fraction=0.15
        )
        with self.assertRaisesRegex(TypeError, re.escape(target_err_msg)):
            cnn.validation_fraction = 0.0

    def test_validation_fraction_negative_3(self):
        """ Проверяем работу свойства validation_fraction при установке слишком большого вещественного значения. """
        target_err_msg = 'Validation fraction is specified incorrectly!'
        cnn = neural_network.CNNClassifier(
            input_maps_number=1, input_map_size=(28, 28),
            structure_of_hidden_layers=(
                {
                    'feature_maps_number': 4,
                    'feature_map_size': (24, 24),
                    'receptive_field_size': (5, 5)
                },
                {
                    'feature_maps_number': 4,
                    'feature_map_size': (12, 12),
                    'receptive_field_size': (2, 2)
                },
                {
                    'feature_maps_number': 12,
                    'feature_map_size': (8, 8),
                    'receptive_field_size': (5, 5)
                },
                {
                    'feature_maps_number': 12,
                    'feature_map_size': (4, 4),
                    'receptive_field_size': (2, 2)
                }
            ),
            classes_number=10,
            max_epochs_number=50,
            learning_rate=0.01,
            early_stopping=True,
            validation_fraction=0.15
        )
        with self.assertRaisesRegex(TypeError, re.escape(target_err_msg)):
            cnn.validation_fraction = 1.0

    def test_fit_positive_1(self):
        """ Проверяем процесс обучения и последующего тестирования СНС при корректно заданном обучающем множестве. """
        cnn = self.create_cnn()
        X_train, y_train = self.create_train_set()
        score_before_training = cnn.score(X_train, y_train)
        cnn.fit(X_train, y_train)
        score_after_training = cnn.score(X_train, y_train)
        self.assertGreater(score_after_training, score_before_training)
        predicted_labels = cnn.predict(X_train)
        self.assertIsInstance(predicted_labels, numpy.ndarray)
        self.assertEqual(predicted_labels.shape, (100,))
        self.assertEqual(predicted_labels.dtype, numpy.uint32)
        for sample_ind in range(predicted_labels.shape[0]):
            self.assertTrue((predicted_labels[sample_ind] >= 0) and (predicted_labels[sample_ind] < 10))

    def test_fit_negative_01(self):
        """ Проверяем генерацию исключения, если входные сигналы обучающего множества - не numpy.ndarray. """
        cnn = self.create_cnn()
        X_train, y_train = self.create_train_set()
        target_err_msg = 'Convolution neural network cannot be trained! Structure of input data does not correspond ' \
                         'to structure of neural network. Input data must be numpy.ndarray.'
        with self.assertRaisesRegex(ECNNTraining, re.escape(target_err_msg)):
            cnn.fit(X_train.tolist(), y_train)

    def test_fit_negative_02(self):
        """ Проверяем генерацию исключения, если во входных сигналах по 2 входных карты, а на входе СНС ожидается одна.
        """
        cnn = self.create_cnn()
        X_train, y_train = self.create_train_set(number_of_input_maps=2)
        target_err_msg = 'Convolution neural network cannot be trained! Structure of input data does not correspond ' \
                         'to structure of neural network. Each input sample must consist of only one input map.'
        with self.assertRaisesRegex(ECNNTraining, re.escape(target_err_msg)):
            cnn.fit(X_train, y_train)

    def test_fit_negative_03(self):
        """ Проверяем генерацию исключения, если во входных сигналах по одной входной карте, а на входе СНС ожидается 2.
        """
        cnn = self.create_cnn(number_of_input_maps=2)
        X_train, y_train = self.create_train_set()
        target_err_msg = 'Convolution neural network cannot be trained! Structure of input data does not correspond ' \
                         'to structure of neural network. Each input sample must consist of 2 input maps.'
        with self.assertRaisesRegex(ECNNTraining, re.escape(target_err_msg)):
            cnn.fit(X_train, y_train)

    def test_fit_negative_04(self):
        """ Проверяем генерацию исключения, если во входных сигналах по одной входной карте, а на входе СНС ожидается 2.
        """
        cnn = self.create_cnn(number_of_input_maps=2)
        X_train, y_train = self.create_train_set(number_of_input_maps=0)
        target_err_msg = 'Convolution neural network cannot be trained! Structure of input data does not correspond ' \
                         'to structure of neural network. Each input sample must consist of 2 input maps.'
        with self.assertRaisesRegex(ECNNTraining, re.escape(target_err_msg)):
            cnn.fit(X_train, y_train)

    def test_fit_negative_05(self):
        """ Проверяем генерацию исключения, если размеры вх.карт в обучающем множестве не совпадают с таковыми в СНС.
        """
        cnn = self.create_cnn()
        X_train, y_train = self.create_train_set(size_of_input_map=(30, 29))
        target_err_msg = 'Convolution neural network cannot be trained! Structure of input data does not correspond ' \
                         'to structure of neural network. Each input map in input data samples must be 28-by-28 matrix.'
        with self.assertRaisesRegex(ECNNTraining, re.escape(target_err_msg)):
            cnn.fit(X_train, y_train)

    def test_fit_negative_06(self):
        """ Проверяем генерацию исключения, если размеры вх.карт в обучающем множестве не совпадают с таковыми в СНС.
        """
        cnn = self.create_cnn()
        X_train, y_train = self.create_train_set(size_of_input_map=(30, 29), number_of_input_maps=0)
        target_err_msg = 'Convolution neural network cannot be trained! Structure of input data does not correspond ' \
                         'to structure of neural network. Each input map in input data samples must be 28-by-28 matrix.'
        with self.assertRaisesRegex(ECNNTraining, re.escape(target_err_msg)):
            cnn.fit(X_train, y_train)

    def test_fit_negative_07(self):
        """ Проверяем генерацию исключения, если матрица входных сигналов обучающего множества слишком многомерна. """
        cnn = self.create_cnn()
        X_train = numpy.random.normal(0.0, 1.0, (100, 2, 3, 28, 28))
        y_train = numpy.random.randint(0, 10, (100,))
        target_err_msg = 'Convolution neural network cannot be trained! Structure of input data does not correspond ' \
                         'to structure of neural network. Input data has too many dimensions.'
        with self.assertRaisesRegex(ECNNTraining, re.escape(target_err_msg)):
            cnn.fit(X_train, y_train)

    def test_fit_negative_08(self):
        """ Проверяем генерацию исключения, если матрица входных сигналов обучающего множества недостаточно многомерна.
        """
        cnn = self.create_cnn()
        X_train = numpy.random.normal(0.0, 1.0, (100, 28 * 28))
        y_train = numpy.random.randint(0, 10, (100,))
        target_err_msg = 'Convolution neural network cannot be trained! Structure of input data does not correspond ' \
                         'to structure of neural network. Input data has too few dimensions.'
        with self.assertRaisesRegex(ECNNTraining, re.escape(target_err_msg)):
            cnn.fit(X_train, y_train)

    def test_fit_negative_09(self):
        """ Проверяем генерацию исключения, если метки классов обучающего множества - не numpy.ndarray. """
        cnn = self.create_cnn()
        X_train, y_train = self.create_train_set()
        target_err_msg = 'Convolution neural network cannot be trained! Target output must be numpy.ndarray.'
        with self.assertRaisesRegex(ECNNTraining, re.escape(target_err_msg)):
            cnn.fit(X_train, y_train.tolist())

    def test_fit_negative_10(self):
        """ Проверяем генерацию исключения, если число меток классов не соответствует числу входных сигналов. """
        cnn = self.create_cnn()
        X_train, y_train = self.create_train_set()
        n = y_train.shape[0]
        target_err_msg = 'Convolution neural network cannot be trained! Number of target outputs does not correspond ' \
                         'to number of input samples.'
        with self.assertRaisesRegex(ECNNTraining, re.escape(target_err_msg)):
            cnn.fit(X_train, y_train[0:(n-1)])

    def test_fit_negative_11(self):
        """ Проверяем генерацию исключения, если метки классов - не целые числа. """
        cnn = self.create_cnn()
        X_train, y_train = self.create_train_set()
        y_train = numpy.random.uniform(0, 10, (X_train.shape[0],))
        target_err_msg = 'Convolution neural network cannot be trained! Each value of target output must be integer.'
        with self.assertRaisesRegex(ECNNTraining, re.escape(target_err_msg)):
            cnn.fit(X_train, y_train)

    def test_fit_negative_12(self):
        """ Проверяем генерацию исключения, если метки классов - не одномерный массив (а двумерный, например). """
        cnn = self.create_cnn()
        X_train, y_train = self.create_train_set()
        y_train = numpy.random.randint(0, 10, (X_train.shape[0], 3))
        target_err_msg = 'Convolution neural network cannot be trained! Target output has too many dimensions.'
        with self.assertRaisesRegex(ECNNTraining, re.escape(target_err_msg)):
            cnn.fit(X_train, y_train)

    def test_fit_negative_13(self):
        """ Проверяем генерацию исключения, если величины меток классов больше числа нейронов в выходном слое СНС. """
        cnn = self.create_cnn()
        X_train, y_train = self.create_train_set()
        y_train = numpy.random.randint(0, 20, (X_train.shape[0],))
        with self.assertRaisesRegex(ECNNTraining, 'Convolution neural network cannot be trained! Target output '
                                                  'for \d+(st|nd|rd|th) sample is incorrect.'):
            cnn.fit(X_train, y_train)


if __name__ == '__main__':
    unittest.main(verbosity=2)
