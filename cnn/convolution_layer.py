# -*- coding: utf-8 -*-
#
# Скрипт ориентирован на использование в Python 3.*
#

import copy
import functools
import itertools
import math
import numpy
import random

import cnn.common

class EConvolutionLayerCreating(cnn.common.ELayerException):
    """ Ошибка генерируется внутри конструктора класса ConvolutionLayer в случае, если параметры
    создаваемого слоя свёртки заданы неверно. """
    def get_base_msg(self):
        base_msg = 'convolution layer cannot be created!'
        layer_str = self.get_layer_id_as_string()
        if len(layer_str) > 0:
            base_msg = '{0} convolution layer cannot be created!'.format(layer_str)
        return base_msg

class EConvolutionLayerCalculating(cnn.common.ELayerException):
    """ Ошибка генерируется в случае, если данные (входные карты), поданные на вход слоя свёртки,
    некорректны, т.е. не соответствуют по своей структуре этому слою свёртки. """
    def get_base_msg(self):
        base_msg = 'Outputs of convolution layer cannot be calculated!'
        layer_str = self.get_layer_id_as_string()
        if len(layer_str) > 0:
            base_msg = 'Outputs of {0} convolution layer cannot be '\
                       'calculated!'.format(layer_str)
        return base_msg

class EConvolutionLayerGradient(cnn.common.ELayerException):
    """ Ошибка генерируется в случае, если невозможно расчитать карты градиентов для слоя свёртки
    на основе уже известных весов и карт градиентов следующего за ним слоя подвыборки (слой
    подвыборки не соответствует слою свёртки). """
    def get_base_msg(self):
        base_msg = 'Gradient of convolution layer cannot be calculated!'
        layer_str = self.get_layer_id_as_string()
        if len(layer_str) > 0:
            base_msg = 'Gradient of {0} convolution layer cannot be '\
                       'calculated!'.format(layer_str)
        return base_msg

class ConvolutionLayer:
    """ Слой свёртки.

    Вычисления в данном слое реализованы согласно статье "Notes on Convolutional Neural Networks"
    http://cogprints.org/5869/1/cnn_tutorial.pdf

    Дополнительные эвристики:
    1) карты слоя свёртки не обязательно являются квадратными, а могут быть и прямоугольными;
    2) каждая карта слоя свёртки всегда связана со всеми входными картами;
    3) рецептивное поле слоя свёртки скользит по входной карте только с единичным шагом как по
    вертикали, так и по горизонтали, тем самым выделяя перекрывающиеся области во входной карте;
    4) рецептивное поле слоя свёртки также не обязательно является квадратным, а может быть и
    прямоугольным;
    5) активационная функция каждого нейрона в каждой карте слоя свёртки - это гиперболический
    тангенс;
    6) рецептивное поле следующего слоя подвыборки (тоже не обязательно квадратное) скользит по
    картам признаков с шагом, равным размерам самого этого поля, таким образом выделяя строго
    соседние, а не перекрывающиеся области.
    """
    def __init__(self, layer_id, input_maps_number, features_maps_number, feature_map_size, \
                 receptive_field_size):
        if layer_id is None:
            raise EConvolutionLayerCreating('Integer identifier of this layer is specified '\
                                            'incorrectly!')
        if not isinstance(layer_id, int):
            raise EConvolutionLayerCreating('Integer identifier of this layer is specified '\
                                            'incorrectly!')
        if layer_id < 1:
            raise EConvolutionLayerCreating('Integer identifier of this layer is specified '\
                                            'incorrectly!')
        if input_maps_number < 1:
            raise EConvolutionLayerCreating(layer_id,
                                            'Structure of input data for this layer (number of '\
                                            'input maps) is specified incorrectly!')
        if (features_maps_number < 1) or (len(feature_map_size) != 2):
            raise EConvolutionLayerCreating(layer_id, 'Feature maps of this layer are '\
                                            'specified incorrectly!')
        if (feature_map_size[0] < 1) or (feature_map_size[1] < 1):
            raise EConvolutionLayerCreating(layer_id, 'Feature maps of this layer are '\
                                            'specified incorrectly!')
        if len(receptive_field_size) != 2:
            raise EConvolutionLayerCreating(layer_id, 'Receptive field of this layer is '\
                                            'specified incorrectly!')
        if (receptive_field_size[0] < 1) or (receptive_field_size[1] < 1):
            raise EConvolutionLayerCreating(layer_id, 'Receptive field of this layer is '\
                                            'specified incorrectly!')
        # Количество входных карт в данных, подаваемых на вход слоя свёртки
        self.__number_of_input_maps = input_maps_number
        # Размер рецептивного поля слоя свёртки
        self.__receptive_field_size = tuple(receptive_field_size)
        # Размер одной карты признаков слоя свёртки
        self.__feature_map_size = tuple(feature_map_size)
        # Список карт признаков слоя свёртки
        self.__feature_maps = [ numpy.zeros(self.__feature_map_size) \
                                for ind in range(features_maps_number) ]
        # Веса (ядра свёртки) всех карт признаков слоя свёртки
        self.__weights_of_feature_maps = list()
        for ft_map_ind in range(features_maps_number):
            self.__weights_of_feature_maps.append(
                [ (numpy.random.random(self.__receptive_field_size) - 0.5) * 2.0 \
                  for inp_map_ind in range(input_maps_number) ]
            )
        # Величины смещений (biases) всех карт признаков слоя свёртки
        self.__biases_of_feature_maps = [ (random.random() - 0.5) * 2.0 \
                                          for ind in range(features_maps_number) ]
        # Карты градиентов для соответствующих карт признаков слоя свёртки
        self.__gradients = [ numpy.zeros(self.__feature_map_size) \
                             for ind in range(features_maps_number)]
        # Карты градиентов, повёрнутые на 180 градусов
        self.__rotated_gradients = [ numpy.zeros(self.__feature_map_size) \
                                     for ind in range(features_maps_number)]
        # Идентификатор слоя (как правило, номер слоя в нейронной сети)
        self.__layer_id = layer_id
        # Вспомогательный список всех возможных пары индексов карты признаков и входной карты
        self.__ft_map_and_inp_map_indexes = list(itertools.product(range(features_maps_number), \
                                                                   range(input_maps_number)))
        # Вспомогательная матрица для вычисления новых значений ядер свёртки
        self.__tmp_matrix = numpy.zeros(self.__receptive_field_size)

    @property
    def feature_map_size(self):
        return self.__feature_map_size

    @property
    def feature_maps_number(self):
        return len(self.__feature_maps)

    @property
    def input_map_size(self):
        return (self.__feature_map_size[0] + self.__receptive_field_size[0] - 1,
                self.__feature_map_size[1] + self.__receptive_field_size[1] - 1)

    @property
    def input_maps_number(self):
        return self.__number_of_input_maps

    @property
    def receptive_field_size(self):
        return self.__receptive_field_size

    @property
    def layer_id(self):
        return self.__layer_id

    @property
    def number_of_trainable_params(self):
        return (self.__receptive_field_size[0] * self.__receptive_field_size[1] \
                * self.__number_of_input_maps + 1) * len(self.__feature_maps)

    @property
    def weights(self):
        return self.__weights_of_feature_maps

    @weights.setter
    def weights(self, values):
        err_msg = 'New values are not weights of {0} convolution layer!'.format(
            cnn.common.integer_to_ordinal(self.__layer_id)
        )
        if len(values) != len(self.__feature_maps):
            raise TypeError(err_msg)
        for weights_of_feature_map in values:
            if len(weights_of_feature_map) != self.__number_of_input_maps:
                raise TypeError(err_msg)
            for conv_cernel in weights_of_feature_map:
                if not isinstance(conv_cernel, numpy.ndarray):
                    raise TypeError(err_msg)
                if conv_cernel.dtype.name != 'float64':
                    raise TypeError(err_msg)
                if conv_cernel.shape != self.__receptive_field_size:
                    raise TypeError(err_msg)
        self.__weights_of_feature_maps = copy.deepcopy(values)

    @property
    def biases(self):
        return self.__biases_of_feature_maps

    @biases.setter
    def biases(self, values):
        err_msg = 'New values are not biases of {0} convolution layer!'.format(
            cnn.common.integer_to_ordinal(self.__layer_id)
        )
        if len(values) != len(self.__feature_maps):
            raise TypeError(err_msg)
        for cur_value in values:
            if not isinstance(cur_value, float):
                raise TypeError(err_msg)
        self.__biases_of_feature_maps = copy.deepcopy(values)

    @property
    def gradients(self):
        return self.__gradients

    def initialize_weights_and_biases(self):
        """ Инициализировать все веса (ядра свёртки) и смещения случайными значениями. """
        features_maps_number = len(self.__feature_maps)
        for ft_map_ind in range(features_maps_number):
            self.__weights_of_feature_maps[ft_map_ind] = [
                (numpy.random.random(self.__receptive_field_size) - 0.5) * 2.0 \
                for inp_map_ind in range(self.__number_of_input_maps)
            ]
        self.__biases_of_feature_maps = [ (random.random() - 0.5) * 2.0 \
                                          for ind in range(features_maps_number) ]

    def calculate_outputs(self, input_maps):
        """ Вычислить выходные карты слоя свёртки для заданных входных данных.

        В ответ на подачу на вход свёрточного слоя входных карт input_maps вычислить выходы карт
        признаков. Перед вычислением проверить допустимость структуры входных карт, которая должна
        соответствовать структуре слоя свёртки.
        """
        self.__check_input_maps_of_this_layer(input_maps)
        for ft_map_ind in range(len(self.__feature_maps)):
            input_maps_and_conv_kernels = list(
                zip(input_maps, self.__weights_of_feature_maps[ft_map_ind])
            )
            for ind in numpy.ndindex(self.__feature_map_size):
                row1 = ind[0]
                col1 = ind[1]
                row2 = row1 + self.__receptive_field_size[0]
                col2 = col1 + self.__receptive_field_size[1]
                output_of_accumulator = functools.reduce(
                    lambda a, b: a + numpy.sum(b[0][row1:row2, col1:col2] * b[1]),
                    input_maps_and_conv_kernels, self.__biases_of_feature_maps[ft_map_ind]
                )
                self.__feature_maps[ft_map_ind][ind] = math.tanh(output_of_accumulator)
        return self.__feature_maps

    def calculate_gradient(self, next_weights, next_gradient, next_receptive_field_size):
        """ Вычислить карты градиентов слоя свёртки.

        Вычислить карты градиентов слоя свёртки, т.е. локальные градиенты нейронов всех карт
        признаков, на основе информации о весах нейронов next_weights, картах градиентов
        next_gradient и размере рецептивного поля следующего за ним слоя подвыборки. Перед
        вычислением проверить соответствие параметров следующего слоя подвыборки и текущего слоя
        свёртки. При этом предполагается, что карты признаков слоя свёртки уже были расчитаны на
        основе некоторых входных карт, поданных на вход этого слоя свёртки (т.е. прямое
        распространение сигнала уже было, а сейчас идёт обратное распространение ошибки).
        """
        self.__check_gradient_maps_of_next_layer(next_weights, next_gradient, \
                                                 next_receptive_field_size)
        matrix_for_upsampling = numpy.ones(next_receptive_field_size)
        ones_matrix = numpy.ones(self.__feature_map_size)
        for ind in range(len(self.__feature_maps)):
            derivatives = ones_matrix - self.__feature_maps[ind] * self.__feature_maps[ind]
            self.__gradients[ind] = derivatives * next_weights[ind] \
                                    * numpy.kron(next_gradient[ind], matrix_for_upsampling)
            self.__rotated_gradients[ind] = numpy.flipud(numpy.fliplr(self.__gradients[ind]))
        return self.__gradients

    def update_weights_and_biases(self, learning_rate, input_maps):
        """ Обновить веса и смещения всех нейронов слоя свёртки.

        После того, как в результате прямого распространения сигнала были расчитаны карты признаков,
        а в результате обратного распространения ошибки - карты градиентов, обновить веса (ядра
        свёртки) и смещения нейронов слоя свёртки. """
        self.__check_input_maps_of_this_layer(input_maps)
        if learning_rate is None:
            return
        for ind in range(len(self.__feature_maps)):
            self.__biases_of_feature_maps[ind] += numpy.sum(self.__gradients[ind]) * learning_rate
        for ft_inp_ind in self.__ft_map_and_inp_map_indexes:
            for ind in numpy.ndindex(self.__receptive_field_size):
                row1 = ind[0]
                col1 = ind[1]
                row2 = row1 + self.__feature_map_size[0]
                col2 = col1 + self.__feature_map_size[1]
                self.__tmp_matrix[ind] = numpy.sum(
                    input_maps[ft_inp_ind[1]][row1:row2, col1:col2] \
                    * self.__rotated_gradients[ft_inp_ind[0]]
                )
            self.__weights_of_feature_maps[ft_inp_ind[0]][ft_inp_ind[1]] += numpy.flipud(\
                numpy.fliplr(self.__tmp_matrix)) * learning_rate

    def __check_input_maps_of_this_layer(self, input_maps):
        """ Проверить корректность структуры входных карт input_maps.

        Сгенерировать исключение EConvolutionLayerCalculating в ситуации, если заданные входные
        карты не являются массивами numpy.ndarray, или же их количество или их размеры не
        соответствуют ожидаемым.
        """
        if input_maps is None:
            raise EConvolutionLayerCalculating(self.__layer_id, 'Input maps are undefined!')
        number_of_input_maps = len(input_maps)
        if number_of_input_maps != self.__number_of_input_maps:
            raise EConvolutionLayerCalculating(self.__layer_id,
                                               'Number of input maps is incorrect!')
        if not isinstance(input_maps[0], numpy.ndarray):
            raise EConvolutionLayerCalculating(self.__layer_id,
                                               'Type of 1st input map is inadmissible!')
        input_size = input_maps[0].shape
        if len(input_size) != 2:
            raise EConvolutionLayerCalculating(self.__layer_id,
                                               '1st input map is not two-dimensional!')
        if (input_size[0] - (self.__receptive_field_size[0] - 1)) != self.__feature_map_size[0]:
            raise EConvolutionLayerCalculating(self.__layer_id,
                                               'Row number of 1st input map is incorrect!')
        if (input_size[1] - (self.__receptive_field_size[1] - 1)) != self.__feature_map_size[1]:
            raise EConvolutionLayerCalculating(self.__layer_id,
                                               'Column number of 1st input map is incorrect!')
        for ind in range(number_of_input_maps-1):
            if not isinstance(input_maps[ind+1], numpy.ndarray):
                raise EConvolutionLayerCalculating(
                    self.__layer_id, 'Type of {0} input map is inadmissible!'.format(
                        cnn.common.integer_to_ordinal(ind+2))
                )
            if input_maps[ind+1].shape != input_maps[ind].shape:
                raise EConvolutionLayerCalculating(
                    self.__layer_id,
                    'Size of {0} input map does not coinside with size of other '\
                    'input maps!'.format(cnn.common.integer_to_ordinal(ind+2))
                )

    def __check_gradient_maps_of_next_layer(self, next_weights, next_gradient_maps, \
                                            next_receptive_field_size):
        """ Сгенерировать исключение EConvolutionLayerGradient, если параметры следующего слоя
        подвыборки заданы некорректно по отношению к текущему слою свёртки.
        """
        if next_weights is None:
            raise EConvolutionLayerGradient(
                self.__layer_id, 'Weights of neuron in next layer are undefined!'
            )
        if next_gradient_maps is None:
            raise EConvolutionLayerGradient(
                self.__layer_id, 'Gradient maps in next layer are undefined!'
            )
        if next_receptive_field_size is None:
            raise EConvolutionLayerGradient(
                self.__layer_id, 'Receptive field size of next layer is undefined!'
            )
        number_of_feature_maps = len(self.__feature_maps)
        if len(next_gradient_maps) != number_of_feature_maps:
            raise EConvolutionLayerGradient(
                self.__layer_id, 'Gradient maps number in next layer does not correspond to '\
                'feature maps number in this layer.'
            )
        if not isinstance(next_gradient_maps[0], numpy.ndarray):
            raise EConvolutionLayerGradient(
                self.__layer_id, 'Type of 1st gradient map in next layer is inadmissible!'
            )
        shape_of_next_layer = next_gradient_maps[0].shape
        if len(shape_of_next_layer) != 2:
            raise EConvolutionLayerGradient(
                self.__layer_id, '1st gradient map in next layer is not two-dimensional!'
            )
        if len(next_receptive_field_size) != 2:
            raise EConvolutionLayerGradient(
                self.__layer_id, 'Receptive field of next layer is not two-dimensional!'
            )
        if (self.__feature_map_size[0] / shape_of_next_layer[0]) != next_receptive_field_size[0]:
            raise EConvolutionLayerGradient(
                self.__layer_id, 'Row number of first gradient map in next layer does not '\
                'correspond to row number of feature maps in this layer.'
            )
        if (self.__feature_map_size[1] / shape_of_next_layer[1]) != next_receptive_field_size[1]:
            raise EConvolutionLayerGradient(
                self.__layer_id, 'Column number of first gradient map in next layer does not '\
                'correspond to column number of feature maps in this layer.'
            )
        for ind in range(number_of_feature_maps-1):
            if not isinstance(next_gradient_maps[ind+1], numpy.ndarray):
                raise EConvolutionLayerGradient(
                    self.__layer_id,
                    'Type of {0} gradient map in next layer is '\
                    'inadmissible!'.format(cnn.common.integer_to_ordinal(ind+2))
                )
            if next_gradient_maps[ind+1].shape != shape_of_next_layer:
                raise EConvolutionLayerGradient(
                    self.__layer_id,
                    'Size of {0} gradient map in next layer does not coinside with size of other '\
                    'gradient maps in that layer!'.format(cnn.common.integer_to_ordinal(ind+2))
                )
        if len(next_weights) != number_of_feature_maps:
            raise EConvolutionLayerGradient(
                self.__layer_id, 'Weigths number in next layer does not correspond to '\
                'feature maps number in this layer.'
            )
        for ind in range(number_of_feature_maps):
            if not isinstance(next_weights[ind], float):
                raise EConvolutionLayerGradient(
                    self.__layer_id,
                    'Type of {0} neuron\'s weight in next (subsampling) layer is '\
                    'inadmissible!'.format(cnn.common.integer_to_ordinal(ind+2))
                )
