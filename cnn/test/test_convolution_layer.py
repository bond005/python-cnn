# -*- coding: utf-8 -*-
#
# Скрипт ориентирован на использование в Python 3.*
# Все данные и расчёты, используемые в позитивных тестах, приведены в гуглодоке:
# https://docs.google.com/spreadsheets/d/1KXuLShbVoL2kaRFOCbTMkS-BNMuc3xVwZA8HwAuShno/edit?usp=sharing
#

import copy
import os
import random
import sys
import numpy
import unittest

cnn_package_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(cnn_package_path)
from cnn import convolution_layer
from cnn.common import integer_to_ordinal

class TestConvolutionLayer(unittest.TestCase):
    def setUp(self):
        self.__epsilon = 0.000001
        random.seed()
        numpy.random.seed()
        self.__layer_id = 1
        self.__number_of_input_maps = 3
        self.__input_maps = [
            numpy.array([
                (-0.14062, 0.293809, 0.905852, -0.45878, 0.740724),
                (-0.68267, -0.09463, 0.614261, -0.50213, 0.565014),
                (0.076374, -0.7649, -0.30093, 0.471437, -0.32848),
                (-0.38347, 0.160011, -0.30884, 0.493158, -0.28132),
                (-0.64146, -0.92638, 0.867563, -0.10696, -0.05661),
                (0.422351, -0.06871, 0.186391, -0.49686, 0.870728)
            ]),
            numpy.array([
                (-0.20349, 0.031203, -0.12173, 0.743632, 0.328677),
                (-0.20938, 0.00954, 0.517926, 0.607911, 0.535574),
                (0.808547, 0.074892, -0.54315, -0.34995, 0.639988),
                (0.773657, -0.29811, -0.13906, -0.51, -0.00329),
                (-0.27878, 0.498802, -0.15015, 0.823873, 0.800871),
                (0.100963, 0.540328, -0.4175, -0.13177, 0.149454)
            ]),
            numpy.array([
                (-0.79188, 0.276618, -0.43735, -0.72712, -0.85333),
                (-0.63524, -0.39295, 0.667738, -0.7939, 0.728149),
                (-0.7309, 0.15348, -0.78023, -0.05637, -0.56361),
                (-0.02899, 0.209677, 0.162097, 0.481749, 0.270865),
                (-0.92895, 0.193228, -0.10883, 0.21654, -0.25176),
                (-0.62173, -0.15969, 0.700956, 0.87735, 0.450899)
            ])
        ]
        self.__feature_map_size = (4, 4)
        self.__number_of_feature_maps = 2
        self.__receptive_field_size = (3, 2)
        self.__weights_before_learning = [
            [
                numpy.array([
                    (0.248911, -0.80479),
                    (-0.05994, 0.444543),
                    (-0.11272, 0.347034)
                ]),
                numpy.array([
                    (-0.4064, 0.903775),
                    (0.459543, 0.013773),
                    (0.030862, -0.82151)
                ]),
                numpy.array([
                    (-0.67126, 0.522565),
                    (0.6297, -0.67527),
                    (0.895411, -0.53919)
                ])
            ],
            [
                numpy.array([
                    (0.3869, -0.70121),
                    (-0.15096, 0.324636),
                    (-0.11414, -0.71626)
                ]),
                numpy.array([
                    (-0.06011, 0.259013),
                    (0.130328, 0.068167),
                    (-0.42851, -0.0707)
                ]),
                numpy.array([
                    (0.755892, -0.13054),
                    (-0.38519, -0.62926),
                    (-0.94621, -0.01596)
                ])
            ]
        ]
        self.__biases_before_learning = [0.771458, -0.05699]
        self.__target_outputs = [
            numpy.array([
                (0.007272564049, 0.1576517872, 0.9917658836, -0.6502549604),
                (0.5948174386, 0.9620815663, 0.6806214495, 0.9014731938),
                (0.3783515592, 0.2948800938, -0.3549596038, 0.7771264012),
                (-0.9704779293, 0.9440387971, -0.2721853578, 0.9888982088)
            ]),
            numpy.array([
                (0.4481509804, -0.1857352404, 0.8783112845, -0.523111819),
                (-0.7574410838, -0.2098844672, 0.8677515255, -0.8144231533),
                (0.9320259303, -0.8679319004, -0.8545958044, -0.5173725551),
                (0.04977667093, 0.5533940627, -0.7461673046, -0.4086499957)
            ])
        ]
        self.__target_gradients = [
            numpy.array([
                (-0.1286712969, -0.125479926, -0.003086782247, -0.1086308843),
                (-0.08315078815, -0.009573529862, -0.1010244168, -0.0352610562),
                (-0.03876598933, -0.04130841688, -0.1998880078, -0.09058377658),
                (-0.002631869866, -0.004921958989, -0.2117603566, -0.0050498571)
            ]),
            numpy.array([
                (0.6823529444, 0.824381655, 0.07685288204, 0.2442252885),
                (0.3639761863, 0.8162241735, 0.08305237473, 0.1132151781),
                (0.06970962401, 0.1309469792, 0.02689007509, 0.07302474399),
                (0.5294916584, 0.3682499105, 0.04419765396, 0.08306412725)
            ])
        ]
        self.__learning_rate = 1.0
        self.__weights_after_learning = [
            [
                numpy.array([
                    (0.2607640631, -0.534899216),
                    (0.09703538085, 0.5432083469),
                    (-0.3830311453, 0.4118434006)
                ]),
                numpy.array([
                    (-0.3527307169, 0.851528009),
                    (0.2064139795, -0.05539638554),
                    (0.001288276516, -0.8581783347)
                ]),
                numpy.array([
                    (-0.7167296398, 0.2965440048),
                    (0.5921727356, -0.4957640816),
                    (0.9013761848, -0.5078213538)
                ])
            ],
            [
                numpy.array([
                    (0.3994403053, -0.1875066265),
                    (-0.03629930338, 0.514195483),
                    (0.5403872099, -0.7596237015)
                ]),
                numpy.array([
                    (1.140775053, -0.2418385847),
                    (1.524280159, 0.7016852301),
                    (-0.4473626788, -0.4116874361)
                ]),
                numpy.array([
                    (1.660369137, 0.4699600389),
                    (0.2229889338, -0.8614060665),
                    (-1.379033413, -0.8964031046)
                ])
            ]
        ]
        self.__biases_after_learning = [-0.4183309135, 4.472865455]
        self.__weights_of_next_subsampling_layer = [-0.24999, 0.935354]
        self.__receptive_field_size_of_next_subsampling_layer = (2, 2)
        self.__biases_of_next_subsampling_layer = [-0.20395, 0.671541]
        self.__gradients_of_next_subsampling_layer = [
            numpy.array([
                (0.514733, 0.752884),
                (0.180977, 0.914852)
            ]),
            numpy.array([
                (0.912849, 0.359473),
                (0.567493, 0.106608)
            ])
        ]
        self.__conv_layer = convolution_layer.ConvolutionLayer(
            self.__layer_id,
            self.__number_of_input_maps,
            self.__number_of_feature_maps,
            self.__feature_map_size,
            self.__receptive_field_size
        )

    def __same_weights_and_biases(self, old_weights, old_biases, new_weights, new_biases):
        is_same = True
        if len(old_weights) == len(new_weights):
            for ft_ind in range(len(old_weights)):
                if len(old_weights[ft_ind]) != len(new_weights[ft_ind]):
                    is_same = False
                else:
                    for inp_ind in range(len(old_weights[ft_ind])):
                        if not isinstance(old_weights[ft_ind][inp_ind], numpy.ndarray):
                            is_same = False
                            break
                        if not isinstance(new_weights[ft_ind][inp_ind], numpy.ndarray):
                            is_same = False
                            break
                        if old_weights[ft_ind][inp_ind].shape != new_weights[ft_ind][inp_ind].shape:
                            is_same = False
                            break
                        for ind in numpy.ndindex(old_weights[ft_ind][inp_ind].shape):
                            deviation = abs(old_weights[ft_ind][inp_ind][ind]\
                                            - new_weights[ft_ind][inp_ind][ind])
                            if deviation > self.__epsilon:
                                is_same = False
                                break
                        if not is_same:
                            break
                if not is_same:
                    break
        else:
            is_same = False
        if is_same:
            if len(old_biases) == len(new_biases):
                for ft_ind in range(len(old_biases)):
                    if abs(old_biases[ft_ind] - new_biases[ft_ind]) > self.__epsilon:
                        is_same = False
                        break
            else:
                is_same = False
        return is_same

    def __initialize_weights_and_biases(self):
        repeats = 0
        is_same = True
        while is_same and (repeats < 10):
            self.__conv_layer.initialize_weights_and_biases()
            is_same = self.__same_weights_and_biases(
                self.__conv_layer.weights, self.__conv_layer.biases,
                self.__weights_before_learning, self.__biases_before_learning
            )
            repeats += 1
        return not is_same

    def test_weights_and_biases_test_positive_1(self):
        """ Проверить, как записываются и читаются свойства weights ('Веса') и biases
        ('Смещения'). """
        self.assertTrue(self.__initialize_weights_and_biases(),
                        msg = 'Weights and biases of convolution layer cannot be initialized!')
        self.__conv_layer.weights = self.__weights_before_learning
        self.__conv_layer.biases = self.__biases_before_learning
        new_weights = self.__conv_layer.weights
        self.assertEqual(len(new_weights), self.__number_of_feature_maps,
                         msg = 'Target {0} != real {1}: number of feature maps with '\
                         'convolution kernels is incorrect!'.format(self.__number_of_feature_maps,
                                                                    len(new_weights))
                         )
        for ft_ind in range(self.__number_of_feature_maps):
            self.assertEqual(len(new_weights[ft_ind]), self.__number_of_input_maps,
                             msg = 'Target {0} != real {1}: number of convolution kernels '\
                             'of {2} feature map is incorrect!'.format(
                                 self.__number_of_input_maps, len(new_weights[ft_ind]),
                                 integer_to_ordinal(ft_ind+1))
                             )
            for inp_ind in range(self.__number_of_input_maps):
                self.assertIsInstance(new_weights[ft_ind][inp_ind], numpy.ndarray,
                                      msg = 'Type of {0} convolution kernel of {1} feature map is '\
                                      'incorrect!'.format(integer_to_ordinal(inp_ind+1),
                                                          integer_to_ordinal(ft_ind+1))
                                      )
                self.assertEqual(new_weights[ft_ind][inp_ind].shape, self.__receptive_field_size,
                                 msg = 'Sizes of {0} convolution kernel of {1} feature map are '\
                                 'incorrect!'.format(integer_to_ordinal(inp_ind+1),
                                                     integer_to_ordinal(ft_ind+1))
                                 )
                for ind in numpy.ndindex(self.__receptive_field_size):
                    self.assertAlmostEqual(
                        new_weights[ft_ind][inp_ind][ind],
                        self.__weights_before_learning[ft_ind][inp_ind][ind],
                        msg = 'Values of {0} convolution kernel of {1} feature map are '\
                        'incorrect!\n{2}'.format(
                            integer_to_ordinal(inp_ind+1), integer_to_ordinal(ft_ind+1),
                            str(new_weights[ft_ind][inp_ind]))
                    )
        new_biases = self.__conv_layer.biases
        self.assertEqual(len(new_biases), self.__number_of_feature_maps,
                         msg = 'Target {0} != real {1}: number of feature maps with '\
                         'biases is incorrect!'.format(self.__number_of_feature_maps,
                                                       len(new_biases))
                         )
        for ft_ind in range(self.__number_of_feature_maps):
            self.assertIsInstance(new_biases[ft_ind], float,
                                  msg = 'Type of {0} feature map\'s bias is incorrect'.format(
                                      integer_to_ordinal(ft_ind+1))
                                  )
            self.assertAlmostEqual(new_biases[ft_ind], self.__biases_before_learning[ft_ind],
                                   msg = 'Target {0} != real {1}: value of {2} feature map\'s bias'\
                                   ' is incorrect'.format(self.__biases_before_learning[ft_ind],
                                                          new_biases[ft_ind],
                                                          integer_to_ordinal(ft_ind+1))
                                   )

    def test_weights_test_negative_1(self):
        """ Веса слоя - это список списков NumPy-матриц. Проверить ситуацию, когда вместо этого
        на вход свойства weights подаётся какой-то отфонарный список списков вещественных чисел. """
        new_weights = [[random.random() for ind in range(3)] for ind in range(4)]
        with self.assertRaises(TypeError):
            self.__conv_layer.weights = new_weights

    def test_weights_test_negative_2(self):
        """ Веса слоя - это список списков NumPy-матриц (количество фичекарт - количество входных
        карт - матрицы ядер свёртки). Проверить ситуацию, когда по количеству фичекарт и количеству
        входных карт всё ок, а вот NumPy-матрицы не соответствуют ожидаемым ядрам свёртки по
        размерам. """
        new_weights = [
            [
                numpy.array([
                    (0.248911, -0.80479, 0.0),
                    (-0.05994, 0.444543, 0.0),
                    (-0.11272, 0.347034, 0.0)
                ]),
                numpy.array([
                    (-0.4064, 0.903775, 0.0),
                    (0.459543, 0.013773, 0.0),
                    (0.030862, -0.82151, 0.0)
                ]),
                numpy.array([
                    (-0.67126, 0.522565, 0.0),
                    (0.6297, -0.67527, 0.0),
                    (0.895411, -0.53919, 0.0)
                ])
            ],
            [
                numpy.array([
                    (0.3869, -0.70121, 0.0),
                    (-0.15096, 0.324636, 0.0),
                    (-0.11414, -0.71626, 0.0)
                ]),
                numpy.array([
                    (-0.06011, 0.259013, 0.0),
                    (0.130328, 0.068167, 0.0),
                    (-0.42851, -0.0707, 0.0)
                ]),
                numpy.array([
                    (0.755892, -0.13054, 0.0),
                    (-0.38519, -0.62926, 0.0),
                    (-0.94621, -0.01596, 0.0)
                ])
            ]
        ]
        with self.assertRaises(TypeError):
            self.__conv_layer.weights = new_weights

    def test_weights_test_negative_3(self):
        """ Веса слоя - это список списков NumPy-матриц (количество фичекарт - количество входных
        карт - матрицы ядер свёртки). Проверить ситуацию, когда по всем размерам всё ок, но вот
        матрицы ядер свёртки - это не NumPy-матрицы, а обычные двумерные списки Python. """
        new_weights = [
            [
                [
                    [0.2607640631, -0.534899216],
                    [0.09703538085, 0.5432083469],
                    [-0.3830311453, 0.4118434006]
                ],
                [
                    [-0.3527307169, 0.851528009],
                    [0.2064139795, -0.05539638554],
                    [0.001288276516, -0.8581783347]
                ],
                [
                    [-0.7167296398, 0.2965440048],
                    [0.5921727356, -0.4957640816],
                    [0.9013761848, -0.5078213538]
                ]
            ],
            [
                [
                    [0.2614513053, -0.2910866265],
                    [0.05472069662, 0.634102483],
                    [0.5418072099, 0.3036702985]
                ],
                [
                    [0.7944850535, 0.4029234153],
                    [1.853495159, 0.6472912301],
                    [0.01200932117, -1.162497436]
                ],
                [
                    [0.2332171371, 1.123065039],
                    [1.237878934, -0.9074160665],
                    [0.4625875868, -1.419633105]
                ]
            ]
        ]
        with self.assertRaises(TypeError):
            self.__conv_layer.weights = new_weights

    def test_biases_test_negative_1(self):
        """ Смещения (biases) слоя - это обычный Python-список вещественных чисел, длина которого
        равна количеству фичемап в слое. А что, если мы попытаемся записать в biases список большей
        длины? """
        new_biases = [random.random() for ind in range(self.__number_of_feature_maps + 2)]
        with self.assertRaises(TypeError):
            self.__conv_layer.biases = new_biases

    def test_biases_test_negative_2(self):
        """ Смещения (biases) слоя - это обычный Python-список вещественных чисел, длина которого
        равна количеству фичемап в слое. А что, если мы не все значения этого списка будут
        вещественными? """
        new_biases = [0.3, 2, 'a', -1]
        with self.assertRaises(TypeError):
            self.__conv_layer.biases = new_biases

    def test_number_of_trainable_params(self):
        """ Проверить, что свойство number_of_trainable_params (количество обучаемых параметров
        слоя) возвращает правильное число. """
        self.assertEqual(self.__conv_layer.number_of_trainable_params,
                         (self.__receptive_field_size[0] * self.__receptive_field_size[1] \
                          * self.__number_of_input_maps + 1) * self.__number_of_feature_maps)

    def test_layer_id(self):
        """ Проверить, что идентификатор слоя установлен правильно, т.е. свойство layer_id
        возвращает именно то число, которое было установлено при создании этого слоя. """
        self.assertEqual(self.__conv_layer.layer_id, self.__layer_id)

    def test_calculate_outputs_test_positive_1(self):
        """ Проверить, что выходы (выходные карты) слоя рассчитываются правильно при подаче на вход
        заданных корректных входных сигналов (набора входных карт). """
        self.__conv_layer.weights = self.__weights_before_learning
        self.__conv_layer.biases = self.__biases_before_learning
        calculated_outputs = self.__conv_layer.calculate_outputs(self.__input_maps)
        self.assertEqual(len(calculated_outputs), self.__number_of_feature_maps,
                         msg = 'Target {0} != real {1}: Number of calculated output maps is '\
                         'incorrect!'.format(self.__number_of_feature_maps,
                                             len(calculated_outputs))
                         )
        for ft_ind in range(self.__number_of_feature_maps):
            self.assertIsInstance(calculated_outputs[ft_ind], numpy.ndarray,
                                  msg = 'Type of output map {0} is incorrect!'.format(ft_ind+1)
                                  )
            self.assertEqual(calculated_outputs[ft_ind].shape, self.__feature_map_size,
                             msg = 'Target {0} != real {1}: Sizes of output map {2} is '\
                             'incorrect!'.format(str(self.__feature_map_size),
                                                 str(calculated_outputs[ft_ind].shape), ft_ind+1)
                             )
            for outp_ind in numpy.ndindex(self.__feature_map_size):
                self.assertAlmostEqual(
                    calculated_outputs[ft_ind][outp_ind],
                    self.__target_outputs[ft_ind][outp_ind],
                    msg = 'Output map {0} is incorrect!\n{1}'.format(
                        ft_ind+1, str(calculated_outputs[ft_ind]))
                )

    def test_calculate_outputs_test_negative_1(self):
        """ Проверить ситуацию, когда входной сигнал некорректен - входные карты правильного
        размера, но их слишком много. """
        input_maps = [
            numpy.array([
                (-0.14062, 0.293809, 0.905852, -0.45878, 0.740724),
                (-0.68267, -0.09463, 0.614261, -0.50213, 0.565014),
                (0.076374, -0.7649, -0.30093, 0.471437, -0.32848),
                (-0.38347, 0.160011, -0.30884, 0.493158, -0.28132),
                (-0.64146, -0.92638, 0.867563, -0.10696, -0.05661),
                (0.422351, -0.06871, 0.186391, -0.49686, 0.870728)
            ]),
            numpy.array([
                (-0.20349, 0.031203, -0.12173, 0.743632, 0.328677),
                (-0.20938, 0.00954, 0.517926, 0.607911, 0.535574),
                (0.808547, 0.074892, -0.54315, -0.34995, 0.639988),
                (0.773657, -0.29811, -0.13906, -0.51, -0.00329),
                (-0.27878, 0.498802, -0.15015, 0.823873, 0.800871),
                (0.100963, 0.540328, -0.4175, -0.13177, 0.149454)
            ]),
            numpy.array([
                (-0.79188, 0.276618, -0.43735, -0.72712, -0.85333),
                (-0.63524, -0.39295, 0.667738, -0.7939, 0.728149),
                (-0.7309, 0.15348, -0.78023, -0.05637, -0.56361),
                (-0.02899, 0.209677, 0.162097, 0.481749, 0.270865),
                (-0.92895, 0.193228, -0.10883, 0.21654, -0.25176),
                (-0.62173, -0.15969, 0.700956, 0.87735, 0.450899)
            ]),
            numpy.array([
                (-0.79188, 0.276618, -0.43735, -0.72712, -0.85333),
                (-0.63524, -0.39295, 0.667738, -0.7939, 0.728149),
                (-0.7309, 0.15348, -0.78023, -0.05637, -0.56361),
                (-0.02899, 0.209677, 0.162097, 0.481749, 0.270865),
                (-0.92895, 0.193228, -0.10883, 0.21654, -0.25176),
                (-0.62173, -0.15969, 0.700956, 0.87735, 0.450899)
            ])
        ]
        with self.assertRaises(convolution_layer.EConvolutionLayerCalculating):
            self.__conv_layer.calculate_outputs(input_maps)

    def test_calculate_outputs_test_negative_2(self):
        """ Проверить ситуацию, когда входной сигнал некорректен - количество входных карт
        верное, но сами они неправильного размера. """
        input_maps = [
            numpy.array([
                (-0.14062, 0.293809, 0.905852, -0.45878, 0.740724),
                (-0.68267, -0.09463, 0.614261, -0.50213, 0.565014),
                (0.076374, -0.7649, -0.30093, 0.471437, -0.32848),
                (-0.38347, 0.160011, -0.30884, 0.493158, -0.28132),
                (-0.64146, -0.92638, 0.867563, -0.10696, -0.05661)
            ]),
            numpy.array([
                (-0.20349, 0.031203, -0.12173, 0.743632, 0.328677),
                (-0.20938, 0.00954, 0.517926, 0.607911, 0.535574),
                (0.808547, 0.074892, -0.54315, -0.34995, 0.639988),
                (0.773657, -0.29811, -0.13906, -0.51, -0.00329),
                (-0.27878, 0.498802, -0.15015, 0.823873, 0.800871)
            ]),
            numpy.array([
                (-0.79188, 0.276618, -0.43735, -0.72712, -0.85333),
                (-0.63524, -0.39295, 0.667738, -0.7939, 0.728149),
                (-0.7309, 0.15348, -0.78023, -0.05637, -0.56361),
                (-0.02899, 0.209677, 0.162097, 0.481749, 0.270865),
                (-0.92895, 0.193228, -0.10883, 0.21654, -0.25176)
            ])
        ]
        with self.assertRaises(convolution_layer.EConvolutionLayerCalculating):
            self.__conv_layer.calculate_outputs(input_maps)

    def test_calculate_outputs_test_negative_3(self):
        """ Проверить ситуацию, когда входной сигнал - ничто. """
        with self.assertRaises(convolution_layer.EConvolutionLayerCalculating):
            self.__conv_layer.calculate_outputs(None)

    def test_calculate_gradient_test_positive_1(self):
        """ Проверить, как расчитываются карты градиентов слоя, когда все входные данные (весовые
        коэффициенты, карты градиентов и размеры рецептивного поля следующего слоя) корректны. """
        self.__conv_layer.weights = self.__weights_before_learning
        self.__conv_layer.biases = self.__biases_before_learning
        self.__conv_layer.calculate_outputs(self.__input_maps)
        self.__conv_layer.calculate_gradient(self.__weights_of_next_subsampling_layer,
                                             self.__gradients_of_next_subsampling_layer,
                                             self.__receptive_field_size_of_next_subsampling_layer)
        calculated_gradients = self.__conv_layer.gradients
        self.assertEqual(len(calculated_gradients), self.__number_of_feature_maps,
                         msg = 'Target {0} != real {1}: Number of calculated output maps is '\
                         'incorrect!'.format(self.__number_of_feature_maps,
                                             len(calculated_gradients))
                         )
        for ft_ind in range(self.__number_of_feature_maps):
            self.assertIsInstance(calculated_gradients[ft_ind], numpy.ndarray,
                                  msg = 'Type of gradient map {0} is incorrect!'.format(ft_ind+1)
                                  )
            self.assertEqual(calculated_gradients[ft_ind].shape, self.__feature_map_size,
                             msg = 'Target {0} != real {1}: Sizes of gradient map {2} is '\
                             'incorrect!'.format(str(self.__feature_map_size),
                                                 str(calculated_gradients[ft_ind].shape), ft_ind+1)
                             )
            for outp_ind in numpy.ndindex(self.__feature_map_size):
                self.assertAlmostEqual(
                    calculated_gradients[ft_ind][outp_ind],
                    self.__target_gradients[ft_ind][outp_ind],
                    msg = 'Gradient map {0} is incorrect!\n{1}'.format(
                        ft_ind+1, str(calculated_gradients[ft_ind]))
                )

    def test_calculate_gradient_test_negative_1(self):
        """ Проверить метод расчёта карт градиентов слоя, когда весовые коэффициенты и карты
        градиентов следующего слоя заданы корректно, а вот вместо размеров рецептивного поля
        следующего слоя задано ничто. """
        with self.assertRaises(convolution_layer.EConvolutionLayerGradient):
            self.__conv_layer.calculate_gradient(self.__weights_of_next_subsampling_layer,
                                                 self.__gradients_of_next_subsampling_layer, None)

    def test_calculate_gradient_test_negative_2(self):
        """ Проверить метод расчёта карт градиентов слоя, когда весовые коэффициенты и размеры
        рецептивного поля следующего слоя заданы корректно, а вот вместо карт градиентов следующего
        слоя задано ничто. """
        with self.assertRaises(convolution_layer.EConvolutionLayerGradient):
            self.__conv_layer.calculate_gradient(
                self.__weights_of_next_subsampling_layer, None,
                self.__receptive_field_size_of_next_subsampling_layer
            )

    def test_calculate_gradient_test_negative_3(self):
        """ Проверить метод расчёта карт градиентов слоя, когда карты градиентов и размеры
        рецептивного поля следующего слоя заданы корректно, а вот вместо весовых коэффициентов
        следующего слоя задано ничто. """
        with self.assertRaises(convolution_layer.EConvolutionLayerGradient):
            self.__conv_layer.calculate_gradient(
                None, self.__gradients_of_next_subsampling_layer,
                self.__receptive_field_size_of_next_subsampling_layer
            )

    def test_calculate_gradient_test_negative_4(self):
        """ Проверить метод расчёта карт градиентов слоя, когда размер рецептивного поля следующего
        слоя задан некорректно (слишком маленький, меньше ожидаемого). """
        with self.assertRaises(convolution_layer.EConvolutionLayerGradient):
            self.__conv_layer.calculate_gradient(self.__weights_of_next_subsampling_layer,
                                                 self.__gradients_of_next_subsampling_layer, (1,1))

    def test_calculate_gradient_test_negative_5(self):
        """ Проверить метод расчёта карт градиентов слоя, когда размер рецептивного поля следующего
        слоя задан некорректно (высота больше ожидаемой). """
        receptive_field_size_of_next_subsampling_layer = (
            self.__receptive_field_size_of_next_subsampling_layer[0] + 1,
            self.__receptive_field_size_of_next_subsampling_layer[1]
        )
        with self.assertRaises(convolution_layer.EConvolutionLayerGradient):
            self.__conv_layer.calculate_gradient(self.__weights_of_next_subsampling_layer,
                                                 self.__gradients_of_next_subsampling_layer,
                                                 receptive_field_size_of_next_subsampling_layer)

    def test_calculate_gradient_test_negative_6(self):
        """ Проверить метод расчёта карт градиентов слоя, когда размер рецептивного поля следующего
        слоя задан некорректно (ширина больше ожидаемой). """
        receptive_field_size_of_next_subsampling_layer = (
            self.__receptive_field_size_of_next_subsampling_layer[0],
            self.__receptive_field_size_of_next_subsampling_layer[1] + 1
        )
        with self.assertRaises(convolution_layer.EConvolutionLayerGradient):
            self.__conv_layer.calculate_gradient(self.__weights_of_next_subsampling_layer,
                                                 self.__gradients_of_next_subsampling_layer,
                                                 receptive_field_size_of_next_subsampling_layer)

    def test_calculate_gradient_test_negative_7(self):
        """ Проверить метод расчёта карт градиентов слоя, когда карты градиентов следующего
        слоя заданы некорректно (карта градиентов является не NumPy-матрицей, а просто двумерным
        Python-списком вещественных чисел). """
        gradients_of_next_subsampling_layer = [
            [
                [-0.1286712969, -0.125479926, -0.003086782247, -0.1086308843],
                [-0.08315078815, -0.009573529862, -0.1010244168, -0.0352610562],
                [-0.03876598933, -0.04130841688, -0.1998880078, -0.09058377658],
                [-0.002631869866, -0.004921958989, -0.2117603566, -0.0050498571]
            ],
            [
                [0.6823529444, 0.824381655, 0.07685288204, 0.2442252885],
                [0.3639761863, 0.8162241735, 0.08305237473, 0.1132151781],
                [0.06970962401, 0.1309469792, 0.02689007509, 0.07302474399],
                [0.5294916584, 0.3682499105, 0.04419765396, 0.08306412725]
            ]
        ]
        with self.assertRaises(convolution_layer.EConvolutionLayerGradient):
            self.__conv_layer.calculate_gradient(
                self.__weights_of_next_subsampling_layer,
                gradients_of_next_subsampling_layer,
                self.__receptive_field_size_of_next_subsampling_layer
            )

    def test_calculate_gradient_test_negative_8(self):
        """ Проверить метод расчёта карт градиентов слоя, когда карты градиентов следующего
        слоя заданы некорректно (карты градиентов действительно являются NumPy-матрицами, но вот
        беда - этих матриц больше, чем надо). """
        gradients_of_next_subsampling_layer = [
            numpy.array([
                (-0.1286712969, -0.125479926, -0.003086782247, -0.1086308843),
                (-0.08315078815, -0.009573529862, -0.1010244168, -0.0352610562),
                (-0.03876598933, -0.04130841688, -0.1998880078, -0.09058377658),
                (-0.002631869866, -0.004921958989, -0.2117603566, -0.0050498571)
            ]),
            numpy.array([
                (0.6823529444, 0.824381655, 0.07685288204, 0.2442252885),
                (0.3639761863, 0.8162241735, 0.08305237473, 0.1132151781),
                (0.06970962401, 0.1309469792, 0.02689007509, 0.07302474399),
                (0.5294916584, 0.3682499105, 0.04419765396, 0.08306412725)
            ]),
            numpy.array([
                (0.6823529444, 0.824381655, 0.07685288204, 0.2442252885),
                (0.3639761863, 0.8162241735, 0.08305237473, 0.1132151781),
                (0.06970962401, 0.1309469792, 0.02689007509, 0.07302474399),
                (0.5294916584, 0.3682499105, 0.04419765396, 0.08306412725)
            ])
        ]
        with self.assertRaises(convolution_layer.EConvolutionLayerGradient):
            self.__conv_layer.calculate_gradient(
                self.__weights_of_next_subsampling_layer,
                gradients_of_next_subsampling_layer,
                self.__receptive_field_size_of_next_subsampling_layer
            )

    def test_calculate_gradient_test_negative_9(self):
        """ Проверить метод расчёта карт градиентов слоя, когда весовые коэффициенты следующего
        слоя заданы некорректно (они действительно являются списком вещественных чисел, но вот длина
        этого списка больше, чем надо). """
        with self.assertRaises(convolution_layer.EConvolutionLayerGradient):
            self.__conv_layer.calculate_gradient(
                [0.0, 0.0, 0.0],
                self.__gradients_of_next_subsampling_layer,
                self.__receptive_field_size_of_next_subsampling_layer
            )

    def test_calculate_gradient_test_negative_10(self):
        """ Проверить метод расчёта карт градиентов слоя, когда весовые коэффициенты следующего
        слоя заданы некорректно (они действительно являются списком нужной длины, но вот не все
        элементы этого списка являются вещественными числами). """
        with self.assertRaises(convolution_layer.EConvolutionLayerGradient):
            self.__conv_layer.calculate_gradient(
                [0.0, 'a'],
                self.__gradients_of_next_subsampling_layer,
                self.__receptive_field_size_of_next_subsampling_layer
            )

    def test_update_weights_and_biases_test_positive_1(self):
        """ Проверить, как обновляются веса и смещения слоя с заданным коэффициентом скорости
        обучения после прямого распространения сигнала и обратного распространения ошибки (т.е.
        выходы и градиенты слоя уже благополучно расчитаны). """
        self.__conv_layer.weights = self.__weights_before_learning
        self.__conv_layer.biases = self.__biases_before_learning
        self.__conv_layer.calculate_outputs(self.__input_maps)
        self.__conv_layer.calculate_gradient(self.__weights_of_next_subsampling_layer,
                                             self.__gradients_of_next_subsampling_layer,
                                             self.__receptive_field_size_of_next_subsampling_layer)
        self.__conv_layer.update_weights_and_biases(self.__learning_rate, self.__input_maps)
        new_weights = self.__conv_layer.weights
        self.assertEqual(len(new_weights), self.__number_of_feature_maps,
                         msg = 'Target {0} != real {1}: number of feature maps with updated '\
                         'convolution kernels is incorrect!'.format(self.__number_of_feature_maps,
                                                                    len(new_weights))
                         )
        for ft_ind in range(self.__number_of_feature_maps):
            self.assertEqual(len(new_weights[ft_ind]), self.__number_of_input_maps,
                             msg = 'Target {0} != real {1}: number of updated convolution kernels '\
                             'of {2} feature map is incorrect!'.format(
                                 self.__number_of_input_maps, len(new_weights[ft_ind]),
                                 integer_to_ordinal(ft_ind+1))
                             )
            for inp_ind in range(self.__number_of_input_maps):
                self.assertIsInstance(new_weights[ft_ind][inp_ind], numpy.ndarray,
                                      msg = 'Type of {0} convolution kernel of {1} feature map is '\
                                      'incorrect!'.format(integer_to_ordinal(inp_ind+1),
                                                          integer_to_ordinal(ft_ind+1))
                                      )
                self.assertEqual(new_weights[ft_ind][inp_ind].shape, self.__receptive_field_size,
                                 msg = 'Sizes of {0} convolution kernel of {1} feature map are '\
                                 'incorrect!'.format(integer_to_ordinal(inp_ind+1),
                                                     integer_to_ordinal(ft_ind+1))
                                 )
                for ind in numpy.ndindex(self.__receptive_field_size):
                    self.assertAlmostEqual(
                        new_weights[ft_ind][inp_ind][ind],
                        self.__weights_after_learning[ft_ind][inp_ind][ind],
                        msg = 'Values of {0} convolution kernel of {1} feature map are '\
                        'incorrect!\n{2}'.format(
                            integer_to_ordinal(inp_ind+1), integer_to_ordinal(ft_ind+1),
                            str(new_weights[ft_ind][inp_ind]))
                    )
        new_biases = self.__conv_layer.biases
        self.assertEqual(len(new_biases), self.__number_of_feature_maps,
                         msg = 'Target {0} != real {1}: number of feature maps with updated '\
                         'biases is incorrect!'.format(self.__number_of_feature_maps,
                                                       len(new_biases))
                         )
        for ft_ind in range(self.__number_of_feature_maps):
            self.assertIsInstance(new_biases[ft_ind], float,
                                  msg = 'Type of {0} feature map\'s bias is incorrect'.format(
                                      integer_to_ordinal(ft_ind+1))
                                  )
            self.assertAlmostEqual(new_biases[ft_ind], self.__biases_after_learning[ft_ind],
                                   msg = 'Target {0} != real {1}: value of {2} feature map\'s bias'\
                                   ' is incorrect'.format(self.__biases_before_learning[ft_ind],
                                                          new_biases[ft_ind],
                                                          integer_to_ordinal(ft_ind+1))
                                   )

    def test_update_weights_and_biases_test_positive_2(self):
        """ Проверить, как обновляются веса и смещения слоя с нулевым коэффициентом скорости
        обучения после прямого распространения сигнала и обратного распространения ошибки (т.е.
        выходы и градиенты слоя уже благополучно расчитаны). Правильный ответ - не меняются. """
        self.__conv_layer.weights = self.__weights_before_learning
        self.__conv_layer.biases = self.__biases_before_learning
        self.__conv_layer.calculate_outputs(self.__input_maps)
        self.__conv_layer.calculate_gradient(self.__weights_of_next_subsampling_layer,
                                             self.__gradients_of_next_subsampling_layer,
                                             self.__receptive_field_size_of_next_subsampling_layer)
        self.__conv_layer.update_weights_and_biases(0.0, self.__input_maps)
        new_weights = self.__conv_layer.weights
        self.assertEqual(len(new_weights), self.__number_of_feature_maps,
                         msg = 'Target {0} != real {1}: number of feature maps with updated '\
                         'convolution kernels is incorrect!'.format(self.__number_of_feature_maps,
                                                                    len(new_weights))
                         )
        for ft_ind in range(self.__number_of_feature_maps):
            self.assertEqual(len(new_weights[ft_ind]), self.__number_of_input_maps,
                             msg = 'Target {0} != real {1}: number of updated convolution kernels '\
                             'of {2} feature map is incorrect!'.format(
                                 self.__number_of_input_maps, len(new_weights[ft_ind]),
                                 integer_to_ordinal(ft_ind+1))
                             )
            for inp_ind in range(self.__number_of_input_maps):
                self.assertIsInstance(new_weights[ft_ind][inp_ind], numpy.ndarray,
                                      msg = 'Type of {0} convolution kernel of {1} feature map is '\
                                      'incorrect!'.format(integer_to_ordinal(inp_ind+1),
                                                          integer_to_ordinal(ft_ind+1))
                                      )
                self.assertEqual(new_weights[ft_ind][inp_ind].shape, self.__receptive_field_size,
                                 msg = 'Sizes of {0} convolution kernel of {1} feature map are '\
                                 'incorrect!'.format(integer_to_ordinal(inp_ind+1),
                                                     integer_to_ordinal(ft_ind+1))
                                 )
                for ind in numpy.ndindex(self.__receptive_field_size):
                    self.assertAlmostEqual(
                        new_weights[ft_ind][inp_ind][ind],
                        self.__weights_before_learning[ft_ind][inp_ind][ind],
                        msg = 'Values of {0} convolution kernel of {1} feature map are '\
                        'incorrect!\n{2}'.format(
                            integer_to_ordinal(inp_ind+1), integer_to_ordinal(ft_ind+1),
                            str(new_weights[ft_ind][inp_ind]))
                    )
        new_biases = self.__conv_layer.biases
        self.assertEqual(len(new_biases), self.__number_of_feature_maps,
                         msg = 'Target {0} != real {1}: number of feature maps with updated '\
                         'biases is incorrect!'.format(self.__number_of_feature_maps,
                                                       len(new_biases))
                         )
        for ft_ind in range(self.__number_of_feature_maps):
            self.assertIsInstance(new_biases[ft_ind], float,
                                  msg = 'Type of {0} feature map\'s bias is incorrect'.format(
                                      integer_to_ordinal(ft_ind+1))
                                  )
            self.assertAlmostEqual(new_biases[ft_ind], self.__biases_before_learning[ft_ind],
                                   msg = 'Target {0} != real {1}: value of {2} feature map\'s bias'\
                                   ' is incorrect'.format(self.__biases_before_learning[ft_ind],
                                                          new_biases[ft_ind],
                                                          integer_to_ordinal(ft_ind+1))
                                   )

    def test_update_weights_and_biases_test_positive_3(self):
        """ Проверить, как обновляются веса и смещения слоя с пустым (None) коэффициентом скорости
        обучения после прямого распространения сигнала и обратного распространения ошибки (т.е.
        выходы и градиенты слоя уже благополучно расчитаны). Правильный ответ - не меняются,
        поскольку пустой коэффициент скорости обучения считается нулевым. """
        self.__conv_layer.weights = self.__weights_before_learning
        self.__conv_layer.biases = self.__biases_before_learning
        self.__conv_layer.calculate_outputs(self.__input_maps)
        self.__conv_layer.calculate_gradient(self.__weights_of_next_subsampling_layer,
                                             self.__gradients_of_next_subsampling_layer,
                                             self.__receptive_field_size_of_next_subsampling_layer)
        self.__conv_layer.update_weights_and_biases(None, self.__input_maps)
        new_weights = self.__conv_layer.weights
        self.assertEqual(len(new_weights), self.__number_of_feature_maps,
                         msg = 'Target {0} != real {1}: number of feature maps with updated '\
                         'convolution kernels is incorrect!'.format(self.__number_of_feature_maps,
                                                                    len(new_weights))
                         )
        for ft_ind in range(self.__number_of_feature_maps):
            self.assertEqual(len(new_weights[ft_ind]), self.__number_of_input_maps,
                             msg = 'Target {0} != real {1}: number of updated convolution kernels '\
                             'of {2} feature map is incorrect!'.format(
                                 self.__number_of_input_maps, len(new_weights[ft_ind]),
                                 integer_to_ordinal(ft_ind+1))
                             )
            for inp_ind in range(self.__number_of_input_maps):
                self.assertIsInstance(new_weights[ft_ind][inp_ind], numpy.ndarray,
                                      msg = 'Type of {0} convolution kernel of {1} feature map is '\
                                      'incorrect!'.format(integer_to_ordinal(inp_ind+1),
                                                          integer_to_ordinal(ft_ind+1))
                                      )
                self.assertEqual(new_weights[ft_ind][inp_ind].shape, self.__receptive_field_size,
                                 msg = 'Sizes of {0} convolution kernel of {1} feature map are '\
                                 'incorrect!'.format(integer_to_ordinal(inp_ind+1),
                                                     integer_to_ordinal(ft_ind+1))
                                 )
                for ind in numpy.ndindex(self.__receptive_field_size):
                    self.assertAlmostEqual(
                        new_weights[ft_ind][inp_ind][ind],
                        self.__weights_before_learning[ft_ind][inp_ind][ind],
                        msg = 'Values of {0} convolution kernel of {1} feature map are '\
                        'incorrect!\n{2}'.format(
                            integer_to_ordinal(inp_ind+1), integer_to_ordinal(ft_ind+1),
                            str(new_weights[ft_ind][inp_ind]))
                    )
        new_biases = self.__conv_layer.biases
        self.assertEqual(len(new_biases), self.__number_of_feature_maps,
                         msg = 'Target {0} != real {1}: number of feature maps with updated '\
                         'biases is incorrect!'.format(self.__number_of_feature_maps,
                                                       len(new_biases))
                         )
        for ft_ind in range(self.__number_of_feature_maps):
            self.assertIsInstance(new_biases[ft_ind], float,
                                  msg = 'Type of {0} feature map\'s bias is incorrect'.format(
                                      integer_to_ordinal(ft_ind+1))
                                  )
            self.assertAlmostEqual(new_biases[ft_ind], self.__biases_before_learning[ft_ind],
                                   msg = 'Target {0} != real {1}: value of {2} feature map\'s bias'\
                                   ' is incorrect'.format(self.__biases_before_learning[ft_ind],
                                                          new_biases[ft_ind],
                                                          integer_to_ordinal(ft_ind+1))
                                   )

    def test_update_weights_and_biases_test_negative_1(self):
        """ Что делать, когда мы хотим обучить слой (обновить веса и смещения) после прямого и
        обратного прохода, а входной сигнал для слоя некорректен - слишком много входных карт. """
        input_maps = [
            numpy.array([
                (-0.14062, 0.293809, 0.905852, -0.45878, 0.740724),
                (-0.68267, -0.09463, 0.614261, -0.50213, 0.565014),
                (0.076374, -0.7649, -0.30093, 0.471437, -0.32848),
                (-0.38347, 0.160011, -0.30884, 0.493158, -0.28132),
                (-0.64146, -0.92638, 0.867563, -0.10696, -0.05661),
                (0.422351, -0.06871, 0.186391, -0.49686, 0.870728)
            ]),
            numpy.array([
                (-0.20349, 0.031203, -0.12173, 0.743632, 0.328677),
                (-0.20938, 0.00954, 0.517926, 0.607911, 0.535574),
                (0.808547, 0.074892, -0.54315, -0.34995, 0.639988),
                (0.773657, -0.29811, -0.13906, -0.51, -0.00329),
                (-0.27878, 0.498802, -0.15015, 0.823873, 0.800871),
                (0.100963, 0.540328, -0.4175, -0.13177, 0.149454)
            ]),
            numpy.array([
                (-0.79188, 0.276618, -0.43735, -0.72712, -0.85333),
                (-0.63524, -0.39295, 0.667738, -0.7939, 0.728149),
                (-0.7309, 0.15348, -0.78023, -0.05637, -0.56361),
                (-0.02899, 0.209677, 0.162097, 0.481749, 0.270865),
                (-0.92895, 0.193228, -0.10883, 0.21654, -0.25176),
                (-0.62173, -0.15969, 0.700956, 0.87735, 0.450899)
            ]),
            numpy.array([
                (-0.79188, 0.276618, -0.43735, -0.72712, -0.85333),
                (-0.63524, -0.39295, 0.667738, -0.7939, 0.728149),
                (-0.7309, 0.15348, -0.78023, -0.05637, -0.56361),
                (-0.02899, 0.209677, 0.162097, 0.481749, 0.270865),
                (-0.92895, 0.193228, -0.10883, 0.21654, -0.25176),
                (-0.62173, -0.15969, 0.700956, 0.87735, 0.450899)
            ])
        ]
        with self.assertRaises(convolution_layer.EConvolutionLayerCalculating):
            self.__conv_layer.update_weights_and_biases(self.__learning_rate, input_maps)

    def test_update_weights_and_biases_test_negative_2(self):
        """ Что делать, когда мы хотим обучить слой (обновить веса и смещения) после прямого и
        обратного прохода, а входной сигнал для слоя некорректен - входных карт столько, сколько
        нужно, но они не такого размера. """
        input_maps = [
            numpy.array([
                (-0.14062, 0.293809, 0.905852, -0.45878, 0.740724),
                (-0.68267, -0.09463, 0.614261, -0.50213, 0.565014),
                (0.076374, -0.7649, -0.30093, 0.471437, -0.32848),
                (-0.38347, 0.160011, -0.30884, 0.493158, -0.28132),
                (-0.64146, -0.92638, 0.867563, -0.10696, -0.05661)
            ]),
            numpy.array([
                (-0.20349, 0.031203, -0.12173, 0.743632, 0.328677),
                (-0.20938, 0.00954, 0.517926, 0.607911, 0.535574),
                (0.808547, 0.074892, -0.54315, -0.34995, 0.639988),
                (0.773657, -0.29811, -0.13906, -0.51, -0.00329),
                (-0.27878, 0.498802, -0.15015, 0.823873, 0.800871)
            ]),
            numpy.array([
                (-0.79188, 0.276618, -0.43735, -0.72712, -0.85333),
                (-0.63524, -0.39295, 0.667738, -0.7939, 0.728149),
                (-0.7309, 0.15348, -0.78023, -0.05637, -0.56361),
                (-0.02899, 0.209677, 0.162097, 0.481749, 0.270865),
                (-0.92895, 0.193228, -0.10883, 0.21654, -0.25176)
            ])
        ]
        with self.assertRaises(convolution_layer.EConvolutionLayerCalculating):
            self.__conv_layer.update_weights_and_biases(self.__learning_rate, input_maps)

    def test_update_weights_and_biases_test_negative_3(self):
        """ Что делать, когда мы хотим обучить слой (обновить веса и смещения) после прямого и
        обратного прохода, а входной сигнал для слоя попросту пуст, ничто, None. """
        with self.assertRaises(convolution_layer.EConvolutionLayerCalculating):
            self.__conv_layer.update_weights_and_biases(self.__learning_rate, None)


if __name__ == '__main__':
    #unittest.main(verbosity=2)
    unittest.main()
