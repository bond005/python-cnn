# -*- coding: utf-8 -*-
#
# Скрипт ориентирован на использование в Python 3.*
# Все данные и расчёты, используемые в позитивных тестах, приведены в гуглодоке:
# https://docs.google.com/spreadsheets/d/1KXuLShbVoL2kaRFOCbTMkS-BNMuc3xVwZA8HwAuShno/edit?usp=sharing
#
import os
import random
import sys
import numpy
import unittest

cnn_package_path = os.path.split(os.path.split(os.getcwd())[0])[0]
sys.path.append(cnn_package_path)
from cnn import subsampling_layer
from cnn.common import integer_to_ordinal

class TestSubsamplingLayer(unittest.TestCase):
    def setUp(self):
        self.__epsilon = 0.000001
        random.seed()
        numpy.random.seed()
        self.__layer_id = 1
        self.__number_of_input_maps = 2
        self.__input_maps = [
            numpy.array([
                (-0.152650, -0.100040, -0.117630, 0.293608, 0.406800, 0.328764),
                (-0.384580, 0.184245, 0.206536, -0.014290, 0.219069, 0.100485),
                (0.030261, 0.335395, 0.027620, 0.419152, 0.144828, -0.381510),
                (0.107437, 0.492572, -0.223060, 0.229728, -0.065340, 0.452209),
                (-0.479650, 0.074179, 0.147958, -0.425410, -0.330670, 0.020440),
                (-0.311970, -0.264390, 0.409592, 0.110323, 0.039960, 0.099275),
                (0.313951, 0.359327, 0.106658, 0.417681, -0.262530, 0.334267),
                (0.268850, -0.397760, -0.433170, 0.291700, 0.323420, -0.388240)
            ]),
            numpy.array([
                (-0.108500, -0.289820, -0.252910, 0.057690, -0.085390, -0.174320),
                (-0.167780, 0.280917, -0.426310, -0.017190, 0.327971, 0.042599),
                (0.354793, 0.175288, 0.470998, 0.142525, 0.321081, -0.171490),
                (-0.459390, -0.284940, 0.333003, 0.393460, 0.293637, -0.402190),
                (-0.478640, -0.352470, 0.135851, 0.271241, 0.149573, 0.062754),
                (-0.338090, 0.190882, 0.018897, 0.159130, 0.025455, 0.290217),
                (-0.184320, -0.020680, -0.074600, 0.397026, 0.434611, -0.471030),
                (-0.314210, 0.001393, 0.471240, -0.270580, 0.458946, 0.317060)
            ])
        ]
        self.__feature_map_size = (4, 3)
        self.__number_of_feature_maps = 2
        self.__receptive_field_size = (2, 2)
        self.__weights_before_learning = [-0.32655, 0.444819]
        self.__biases_before_learning = [0.032391, 0.038402]
        self.__target_outputs = [
            numpy.array([
                (-0.02776706522, -0.06340153405, -0.1001130491),
                (-0.1277564318, -0.1041045345, -0.114769907),
                (0.008167665919, -0.1010155561, -0.00002725124999),
                (-0.08474349259, -0.1036293617, -0.07661346136)
            ]),
            numpy.array([
                (0.1619214174, 0.06397610962, 0.1822313649),
                (0.1937405252, 0.2429538367, 0.1792666923),
                (0.1226887285, 0.1577272981, 0.1659470533),
                (0.039001839, 0.2430551263, 0.2379027867)
            ])
        ]
        self.__target_gradients = [
            numpy.array([
                (-0.08582690834, 0.2268520607, 0.1776888803),
                (0.0380044824, 0.04187582332, -0.110107481),
                (-0.3434222656, -0.01611861735, -0.2483066196),
                (0.1633282823, -0.2608100462, -0.04330379265)
            ]),
            numpy.array([
                (0.1254424326, 0.006265172503, -0.1386492782),
                (0.1541471703, 0.7260977216, -0.09298807092),
                (0.2016896913, -0.185401178, 0.4880214163),
                (0.4567907443, -0.09138750679, 0.0566670354)
            ])
        ]
        self.__learning_rate = 1.0
        self.__weights_after_learning = [-0.338442662, 0.9151912283]
        self.__biases_after_learning = [-0.4277552018, 1.745097351]
        self.__weights_of_next_convolution_layer = [
            [
                numpy.array([
                    (-0.10001, 0.22843),
                    (0.05714, -0.03969)
                ]),
                numpy.array([
                    (-0.1971, 0.175462),
                    (0.498516, -0.42573)
                ])
            ],
            [
                numpy.array([
                    (0.480625, 0.036891),
                    (-0.32014, 0.386875)
                ]),
                numpy.array([
                    (-0.3658, -0.03171),
                    (-0.47089, -0.46745)
                ])
            ],
            [
                numpy.array([
                    (0.400792, 0.258218),
                    (0.28257, -0.03486)
                ]),
                numpy.array([
                    (0.184743, -0.40769),
                    (0.121343, 0.307785)
                ])
            ]
        ]
        self.__receptive_field_size_of_next_convolution_layer = (2, 2)
        self.__biases_of_next_convolution_layer = [0.081794, -0.10958, 0.456785]
        self.__gradients_of_next_convolution_layer = [
            numpy.array([
                (0.205001, 0.264354),
                (0.370529, -0.45901),
                (0.499641, -0.23889)
            ]),
            numpy.array([
                (-0.33944, -0.066),
                (0.01034, -0.24663),
                (-0.44996, -0.17296)
            ]),
            numpy.array([
                (0.243898, 0.470673),
                (-0.29586, 0.212249),
                (-0.02863, -0.39796)
            ])
        ]
        self.__subs_layer = subsampling_layer.SubsamplingLayer(
            self.__layer_id,
            self.__number_of_feature_maps,
            self.__feature_map_size,
            self.__receptive_field_size
        )

    def __same_weights_and_biases(self, old_weights, old_biases, new_weights, new_biases):
        is_same = True
        if len(old_weights) == len(new_weights):
            for ft_ind in range(len(old_weights)):
                if abs(old_weights[ft_ind] - new_weights[ft_ind]) > self.__epsilon:
                    is_same = False
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
            self.__subs_layer.initialize_weights_and_biases()
            is_same = self.__same_weights_and_biases(
                self.__subs_layer.weights, self.__subs_layer.biases,
                self.__weights_before_learning, self.__biases_before_learning
            )
            repeats += 1
        return not is_same

    def test_weights_and_biases_test_positive_1(self):
        """ Проверить, как записываются и читаются свойства weights ('Веса') и biases
        ('Смещения'). """
        self.assertTrue(self.__initialize_weights_and_biases(),
                        msg = 'Weights and biases of convolution layer cannot be initialized!')
        self.__subs_layer.weights = self.__weights_before_learning
        self.__subs_layer.biases = self.__biases_before_learning
        new_weights = self.__subs_layer.weights
        self.assertEqual(len(new_weights), self.__number_of_feature_maps,
                         msg = 'Target {0} != real {1}: number of feature maps with '\
                         'convolution kernels is incorrect!'.format(self.__number_of_feature_maps,
                                                                    len(new_weights))
                         )
        for ft_ind in range(self.__number_of_feature_maps):
            self.assertIsInstance(new_weights[ft_ind], float,
                                  msg = 'Type of {0} feature map\'s weights is incorrect'.format(
                                      integer_to_ordinal(ft_ind+1))
                                  )
            self.assertAlmostEqual(new_weights[ft_ind], self.__weights_before_learning[ft_ind],
                                   msg = 'Target {0} != real {1}: value of {2} feature map\'s '\
                                   'weights is incorrect'.format(
                                       self.__weights_before_learning[ft_ind], new_weights[ft_ind],
                                       integer_to_ordinal(ft_ind+1))
                                   )
        new_biases = self.__subs_layer.biases
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
        """ Веса (weights) слоя - это обычный Python-список вещественных чисел, длина которого
        равна количеству фичемап в слое. А что, если мы попытаемся записать в weights список большей
        длины? """
        new_weights = [random.random() for ind in range(self.__number_of_feature_maps + 2)]
        with self.assertRaises(TypeError):
            self.__subs_layer.weights = new_weights

    def test_weights_test_negative_2(self):
        """ Веса (weights) слоя - это обычный Python-список вещественных чисел, длина которого
        равна количеству фичемап в слое. А что, если мы не все значения этого списка будут
        вещественными? """
        new_weights = [0.3, 2, 'a', -1]
        with self.assertRaises(TypeError):
            self.__subs_layer.weights = new_weights

    def test_biases_test_negative_1(self):
        """ Смещения (biases) слоя - это обычный Python-список вещественных чисел, длина которого
        равна количеству фичемап в слое. А что, если мы попытаемся записать в biases список большей
        длины? """
        new_biases = [random.random() for ind in range(self.__number_of_feature_maps + 2)]
        with self.assertRaises(TypeError):
            self.__subs_layer.biases = new_biases

    def test_biases_test_negative_2(self):
        """ Смещения (biases) слоя - это обычный Python-список вещественных чисел, длина которого
        равна количеству фичемап в слое. А что, если мы не все значения этого списка будут
        вещественными? """
        new_biases = [0.3, 2, 'a', -1]
        with self.assertRaises(TypeError):
            self.__subs_layer.biases = new_biases

    def test_number_of_trainable_params(self):
        """ Проверить, что свойство number_of_trainable_params (количество обучаемых параметров
        слоя) возвращает правильное число. """
        self.assertEqual(self.__subs_layer.number_of_trainable_params,
                         self.__number_of_feature_maps * 2)

    def test_layer_id(self):
        """ Проверить, что идентификатор слоя установлен правильно, т.е. свойство layer_id
        возвращает именно то число, которое было установлено при создании этого слоя. """
        self.assertEqual(self.__subs_layer.layer_id, self.__layer_id)

    def test_calculate_outputs_test_positive_1(self):
        """ Проверить, что выходы (выходные карты) слоя рассчитываются правильно при подаче на вход
        заданных корректных входных сигналов (набора входных карт). """
        self.__subs_layer.weights = self.__weights_before_learning
        self.__subs_layer.biases = self.__biases_before_learning
        calculated_outputs = self.__subs_layer.calculate_outputs(self.__input_maps)
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
        """ Проверить ситуацию, когда надо рассчитать выходы слоя, а входной сигнал
        некорректен - входных карт нужное число, но сами они несоответствующего размера. """
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
            ])
        ]
        with self.assertRaises(subsampling_layer.ESubsamplingLayerCalculating):
            self.__subs_layer.calculate_outputs(input_maps)

    def test_calculate_outputs_test_negative_2(self):
        """ Проверить ситуацию, когда надо рассчитать выходы слоя, а входной сигнал
        некорректен - входные карты правильного размера, но их слишком много. """
        input_maps = [
            numpy.array([
                (-0.152650, -0.100040, -0.117630, 0.293608, 0.406800, 0.328764),
                (-0.384580, 0.184245, 0.206536, -0.014290, 0.219069, 0.100485),
                (0.030261, 0.335395, 0.027620, 0.419152, 0.144828, -0.381510),
                (0.107437, 0.492572, -0.223060, 0.229728, -0.065340, 0.452209),
                (-0.479650, 0.074179, 0.147958, -0.425410, -0.330670, 0.020440),
                (-0.311970, -0.264390, 0.409592, 0.110323, 0.039960, 0.099275),
                (0.313951, 0.359327, 0.106658, 0.417681, -0.262530, 0.334267),
                (0.268850, -0.397760, -0.433170, 0.291700, 0.323420, -0.388240)
            ]),
            numpy.array([
                (-0.108500, -0.289820, -0.252910, 0.057690, -0.085390, -0.174320),
                (-0.167780, 0.280917, -0.426310, -0.017190, 0.327971, 0.042599),
                (0.354793, 0.175288, 0.470998, 0.142525, 0.321081, -0.171490),
                (-0.459390, -0.284940, 0.333003, 0.393460, 0.293637, -0.402190),
                (-0.478640, -0.352470, 0.135851, 0.271241, 0.149573, 0.062754),
                (-0.338090, 0.190882, 0.018897, 0.159130, 0.025455, 0.290217),
                (-0.184320, -0.020680, -0.074600, 0.397026, 0.434611, -0.471030),
                (-0.314210, 0.001393, 0.471240, -0.270580, 0.458946, 0.317060)
            ]),
            numpy.array([
                (-0.108500, -0.289820, -0.252910, 0.057690, -0.085390, -0.174320),
                (-0.167780, 0.280917, -0.426310, -0.017190, 0.327971, 0.042599),
                (0.354793, 0.175288, 0.470998, 0.142525, 0.321081, -0.171490),
                (-0.459390, -0.284940, 0.333003, 0.393460, 0.293637, -0.402190),
                (-0.478640, -0.352470, 0.135851, 0.271241, 0.149573, 0.062754),
                (-0.338090, 0.190882, 0.018897, 0.159130, 0.025455, 0.290217),
                (-0.184320, -0.020680, -0.074600, 0.397026, 0.434611, -0.471030),
                (-0.314210, 0.001393, 0.471240, -0.270580, 0.458946, 0.317060)
            ])
        ]
        with self.assertRaises(subsampling_layer.ESubsamplingLayerCalculating):
            self.__subs_layer.calculate_outputs(input_maps)

    def test_calculate_outputs_test_negative_3(self):
        """ Проверить ситуацию, когда надо рассчитать выходы слоя, а входной сигнал оказался вообще
        None. """
        with self.assertRaises(subsampling_layer.ESubsamplingLayerCalculating):
            self.__subs_layer.calculate_outputs(None)

    def test_calculate_gradient_test_positive_1(self):
        """ Проверить, как расчитываются карты градиентов слоя, когда все входные данные (весовые
        коэффициенты, карты градиентов и размеры рецептивного поля следующего слоя) корректны. """
        self.__subs_layer.weights = self.__weights_before_learning
        self.__subs_layer.biases = self.__biases_before_learning
        self.__subs_layer.calculate_outputs(self.__input_maps)
        self.__subs_layer.calculate_gradient(self.__weights_of_next_convolution_layer,
                                             self.__gradients_of_next_convolution_layer,
                                             self.__receptive_field_size_of_next_convolution_layer)
        calculated_gradients = self.__subs_layer.gradients
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
        with self.assertRaises(subsampling_layer.ESubsamplingLayerGradient):
            self.__subs_layer.calculate_gradient(self.__weights_of_next_convolution_layer,
                                                 self.__gradients_of_next_convolution_layer, None)

    def test_calculate_gradient_test_negative_2(self):
        """ Проверить метод расчёта карт градиентов слоя, когда весовые коэффициенты и размеры
        рецептивного поля следующего слоя заданы корректно, а вот вместо карт градиентов следующего
        слоя задано ничто. """
        with self.assertRaises(subsampling_layer.ESubsamplingLayerGradient):
            self.__subs_layer.calculate_gradient(
                self.__weights_of_next_convolution_layer, None,
                self.__receptive_field_size_of_next_convolution_layer
            )

    def test_calculate_gradient_test_negative_3(self):
        """ Проверить метод расчёта карт градиентов слоя, когда карты градиентов и размеры
        рецептивного поля следующего слоя заданы корректно, а вот вместо весовых коэффициентов
        следующего слоя задано ничто. """
        with self.assertRaises(subsampling_layer.ESubsamplingLayerGradient):
            self.__subs_layer.calculate_gradient(
                None, self.__gradients_of_next_convolution_layer,
                self.__receptive_field_size_of_next_convolution_layer
            )

    def test_calculate_gradient_test_negative_4(self):
        """ Проверить метод расчёта карт градиентов слоя, когда размер рецептивного поля следующего
        слоя задан некорректно (слишком маленький, меньше ожидаемого). """
        with self.assertRaises(subsampling_layer.ESubsamplingLayerGradient):
            self.__subs_layer.calculate_gradient(self.__weights_of_next_convolution_layer,
                                                 self.__gradients_of_next_convolution_layer, (1,1))

    def test_calculate_gradient_test_negative_5(self):
        """ Проверить метод расчёта карт градиентов слоя, когда размер рецептивного поля следующего
        слоя задан некорректно (высота больше ожидаемой). """
        receptive_field_size_of_next_convolution_layer = (
            self.__receptive_field_size_of_next_convolution_layer[0] + 1,
            self.__receptive_field_size_of_next_convolution_layer[1]
        )
        with self.assertRaises(subsampling_layer.ESubsamplingLayerGradient):
            self.__subs_layer.calculate_gradient(self.__weights_of_next_convolution_layer,
                                                 self.__gradients_of_next_convolution_layer,
                                                 receptive_field_size_of_next_convolution_layer)

    def test_calculate_gradient_test_negative_6(self):
        """ Проверить метод расчёта карт градиентов слоя, когда размер рецептивного поля следующего
        слоя задан некорректно (ширина больше ожидаемой). """
        receptive_field_size_of_next_convolution_layer = (
            self.__receptive_field_size_of_next_convolution_layer[0],
            self.__receptive_field_size_of_next_convolution_layer[1] + 1
        )
        with self.assertRaises(subsampling_layer.ESubsamplingLayerGradient):
            self.__subs_layer.calculate_gradient(self.__weights_of_next_convolution_layer,
                                                 self.__gradients_of_next_convolution_layer,
                                                 receptive_field_size_of_next_convolution_layer)

    def test_calculate_gradient_test_negative_7(self):
        """ Проверить метод расчёта карт градиентов слоя, когда карты градиентов следующего
        слоя заданы некорректно (карта градиентов является не NumPy-матрицей, а просто двумерным
        Python-списком вещественных чисел). """
        gradients_of_next_convolution_layer = [
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
        with self.assertRaises(subsampling_layer.ESubsamplingLayerGradient):
            self.__subs_layer.calculate_gradient(
                self.__weights_of_next_convolution_layer,
                gradients_of_next_convolution_layer,
                self.__receptive_field_size_of_next_convolution_layer
            )

    def test_calculate_gradient_test_negative_8(self):
        """ Проверить метод расчёта карт градиентов слоя, когда карты градиентов следующего
        слоя заданы некорректно (карты градиентов действительно являются NumPy-матрицами, но вот
        беда - этих матриц больше, чем надо). """
        gradients_of_next_convolution_layer = [
            numpy.array([
                (0.205001, 0.264354),
                (0.370529, -0.45901),
                (0.499641, -0.23889)
            ]),
            numpy.array([
                (-0.33944, -0.066),
                (0.01034, -0.24663),
                (-0.44996, -0.17296)
            ]),
            numpy.array([
                (0.243898, 0.470673),
                (-0.29586, 0.212249),
                (-0.02863, -0.39796)
            ]),
            numpy.array([
                (0.243898, 0.470673),
                (-0.29586, 0.212249),
                (-0.02863, -0.39796)
            ])
        ]
        with self.assertRaises(subsampling_layer.ESubsamplingLayerGradient):
            self.__subs_layer.calculate_gradient(
                self.__weights_of_next_convolution_layer,
                gradients_of_next_convolution_layer,
                self.__receptive_field_size_of_next_convolution_layer
            )

    def test_calculate_gradient_test_negative_9(self):
        """ Проверить метод расчёта карт градиентов слоя, когда весовые коэффициенты следующего
        слоя заданы некорректно (они являются списком вещественных чисел, а должны быть - списком
        списков NumPy-матриц, ведь следующий слой - это слой свёртки). """
        with self.assertRaises(subsampling_layer.ESubsamplingLayerGradient):
            self.__subs_layer.calculate_gradient(
                [0.0, 0.0, 0.0, 0.0, 0.0],
                self.__gradients_of_next_convolution_layer,
                self.__receptive_field_size_of_next_convolution_layer
            )

    def test_calculate_gradient_test_negative_10(self):
        """ Проверить метод расчёта карт градиентов слоя, когда весовые коэффициенты следующего
        слоя заданы некорректно: они должны быть списком списков NumPy-матриц (ведь следующий
        слой - это слой свёртки), а являются вообще какой-то фигнёй. """
        with self.assertRaises(subsampling_layer.ESubsamplingLayerGradient):
            self.__subs_layer.calculate_gradient(
                [0.0, 'a'],
                self.__gradients_of_next_convolution_layer,
                self.__receptive_field_size_of_next_convolution_layer
            )

    def test_update_weights_and_biases_test_positive_1(self):
        """ Проверить, как обновляются веса и смещения слоя с заданным коэффициентом скорости
        обучения после прямого распространения сигнала и обратного распространения ошибки (т.е.
        выходы и градиенты слоя уже благополучно расчитаны). """
        self.__subs_layer.weights = self.__weights_before_learning
        self.__subs_layer.biases = self.__biases_before_learning
        self.__subs_layer.calculate_outputs(self.__input_maps)
        self.__subs_layer.calculate_gradient(self.__weights_of_next_convolution_layer,
                                             self.__gradients_of_next_convolution_layer,
                                             self.__receptive_field_size_of_next_convolution_layer)
        self.__subs_layer.update_weights_and_biases(self.__learning_rate)
        new_weights = self.__subs_layer.weights
        self.assertEqual(len(new_weights), self.__number_of_feature_maps,
                         msg = 'Target {0} != real {1}: number of feature maps with updated '\
                         'weights is incorrect!'.format(self.__number_of_feature_maps,
                                                        len(new_weights))
                         )
        for ft_ind in range(self.__number_of_feature_maps):
            self.assertIsInstance(new_weights[ft_ind], float,
                                  msg = 'Type of {0} feature map\'s weight is incorrect'.format(
                                      integer_to_ordinal(ft_ind+1))
                                  )
            self.assertAlmostEqual(new_weights[ft_ind], self.__weights_after_learning[ft_ind],
                                   msg = 'Target {0} != real {1}: value of {2} feature map\'s '\
                                   'weight is incorrect'.format(
                                       self.__weights_before_learning[ft_ind], new_weights[ft_ind],
                                       integer_to_ordinal(ft_ind+1))
                                   )
        new_biases = self.__subs_layer.biases
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
        self.__subs_layer.weights = self.__weights_before_learning
        self.__subs_layer.biases = self.__biases_before_learning
        self.__subs_layer.calculate_outputs(self.__input_maps)
        self.__subs_layer.calculate_gradient(self.__weights_of_next_convolution_layer,
                                             self.__gradients_of_next_convolution_layer,
                                             self.__receptive_field_size_of_next_convolution_layer)
        self.__subs_layer.update_weights_and_biases(0.0)
        new_weights = self.__subs_layer.weights
        self.assertEqual(len(new_weights), self.__number_of_feature_maps,
                         msg = 'Target {0} != real {1}: number of feature maps with updated '\
                         'weights is incorrect!'.format(self.__number_of_feature_maps,
                                                        len(new_weights))
                         )
        for ft_ind in range(self.__number_of_feature_maps):
            self.assertIsInstance(new_weights[ft_ind], float,
                                  msg = 'Type of {0} feature map\'s weight is incorrect'.format(
                                      integer_to_ordinal(ft_ind+1))
                                  )
            self.assertAlmostEqual(new_weights[ft_ind], self.__weights_before_learning[ft_ind],
                                   msg = 'Target {0} != real {1}: value of {2} feature map\'s '\
                                   'weight is incorrect'.format(
                                       self.__weights_before_learning[ft_ind], new_weights[ft_ind],
                                       integer_to_ordinal(ft_ind+1))
                                   )
        new_biases = self.__subs_layer.biases
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
        self.__subs_layer.weights = self.__weights_before_learning
        self.__subs_layer.biases = self.__biases_before_learning
        self.__subs_layer.calculate_outputs(self.__input_maps)
        self.__subs_layer.calculate_gradient(self.__weights_of_next_convolution_layer,
                                             self.__gradients_of_next_convolution_layer,
                                             self.__receptive_field_size_of_next_convolution_layer)
        self.__subs_layer.update_weights_and_biases(None)
        new_weights = self.__subs_layer.weights
        self.assertEqual(len(new_weights), self.__number_of_feature_maps,
                         msg = 'Target {0} != real {1}: number of feature maps with updated '\
                         'weights is incorrect!'.format(self.__number_of_feature_maps,
                                                        len(new_weights))
                         )
        for ft_ind in range(self.__number_of_feature_maps):
            self.assertIsInstance(new_weights[ft_ind], float,
                                  msg = 'Type of {0} feature map\'s weight is incorrect'.format(
                                      integer_to_ordinal(ft_ind+1))
                                  )
            self.assertAlmostEqual(new_weights[ft_ind], self.__weights_before_learning[ft_ind],
                                   msg = 'Target {0} != real {1}: value of {2} feature map\'s '\
                                   'weight is incorrect'.format(
                                       self.__weights_before_learning[ft_ind], new_weights[ft_ind],
                                       integer_to_ordinal(ft_ind+1))
                                   )
        new_biases = self.__subs_layer.biases
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


if __name__ == '__main__':
    #unittest.main(verbosity=2)
    unittest.main()
