# -*- coding: utf-8 -*-
#
# Скрипт ориентирован на использование в Python 3.*
#

import copy
import numpy
import random
import cnn.common


class ESubsamplingLayerCreating(cnn.common.ELayerException):
    """ Ошибка генерируется внутри конструктора класса SubsamplingLayer в случае, если параметры
    создаваемого слоя подвыборки заданы неверно. """
    def get_base_msg(self):
        base_msg = 'subsampling layer cannot be created!'
        layer_str = self.get_layer_id_as_string()
        if len(layer_str) > 0:
            base_msg = '{0} subsampling layer cannot be created!'.format(layer_str)
        return base_msg

class ESubsamplingLayerCalculating(cnn.common.ELayerException):
    """ Ошибка генерируется в случае, если данные (входные карты), поданные на вход слоя
    подвыборки, некорректны, т.е. не соответствуют по своей структуре этому слою подвыборки. """
    def get_base_msg(self):
        base_msg = 'Outputs of subsampling layer cannot be calculated!'
        layer_str = self.get_layer_id_as_string()
        if len(layer_str) > 0:
            base_msg = 'Outputs of {0} subsampling layer cannot be '\
                       'calculated!'.format(layer_str)
        return base_msg

class ESubsamplingLayerGradient(cnn.common.ELayerException):
    """ Ошибка генерируется в случае, если невозможно расчитать карты градиентов для слоя подвыборки
    на основе уже известных весов и карт градиентов следующего за ним слоя свёркти (слой свёртки
    не соответствует слою подвыборки). """
    def get_base_msg(self):
        base_msg = 'Gradient of subsampling layer cannot be calculated!'
        layer_str = self.get_layer_id_as_string()
        if len(layer_str) > 0:
            base_msg = 'Gradient of {0} subsampling layer cannot be '\
                       'calculated!'.format(layer_str)
        return base_msg


class SubsamplingLayer:
    """ Слой подвыборки.

    Вычисления в данном слое реализованы согласно статье "Notes on Convolutional Neural Networks"
    http://cogprints.org/5869/1/cnn_tutorial.pdf

    Дополнительные эвристики:
    1) карты слоя подвыборки не обязательно являются квадратными, а могут быть и прямоугольными;
    2) каждая карта слоя подвыборки всегда связана только с одной входной картой - соответствующей
    картой признаков предыдущего слоя свёртки;
    3) рецептивное поле слоя подвыборки скользит по входной карте только с шагом, равным размерам
    самого этого поля, таким образом выделяя строго соседние, а не перекрывающиеся области;
    4) рецептивное поле слоя подвыборки также не обязательно является квадратным, а может быть и
    прямоугольным;
    5) если размер рецептивного поля слоя подвыборки равен N строк на M столбцов, то размер карты
    признаков слоя подвыборки меньше размера карты признаков предыдущего слоя свёртки ровно в N раз
    по высоте и в M раз по ширине;
    6) активационная функция каждого нейрона в каждой карте слоя подвыборки - это гиперболический
    тангенс.
    """
    def __init__(self, layer_id, features_maps_number, feature_map_size, receptive_field_size):
        if layer_id is None:
            raise ESubsamplingLayerCreating('Integer identifier of this layer is specified '\
                                            'incorrectly!')
        if not isinstance(layer_id, int):
            raise ESubsamplingLayerCreating('Integer identifier of this layer is specified '\
                                            'incorrectly!')
        if layer_id < 1:
            raise ESubsamplingLayerCreating('Integer identifier of this layer is specified '\
                                            'incorrectly!')
        if (features_maps_number < 1) or (len(feature_map_size) != 2):
            raise ESubsamplingLayerCreating(layer_id, 'Feature maps of this layer are '\
                                            'specified incorrectly!')
        if (feature_map_size[0] < 1) or (feature_map_size[1] < 1):
            raise ESubsamplingLayerCreating(layer_id, 'Feature maps of this layer are '\
                                            'specified incorrectly!')
        if len(receptive_field_size) != 2:
            raise ESubsamplingLayerCreating(layer_id, 'Receptive field of this layer is '\
                                            'specified incorrectly!')
        if (receptive_field_size[0] < 1) or (receptive_field_size[1] < 1):
            raise ESubsamplingLayerCreating(layer_id, 'Receptive field of this layer is '\
                                            'specified incorrectly!')
        # Размер рецептивного поля слоя подвыборки
        self.__receptive_field_size = tuple(receptive_field_size)
        # Размер одной карты признаков слоя подвыборки
        self.__feature_map_size = tuple(feature_map_size)
        # Список карт признаков слоя подвыборки
        self.__feature_maps = [ numpy.zeros(self.__feature_map_size) \
                                for ind in range(features_maps_number) ]
        # Величины весов всех карт признаков слоя подвыборки (для каждой карты - один вес)
        self.__weights_of_feature_maps = [ (random.random() - 0.5) * 2.0 \
                                           for ind in range(features_maps_number) ]
        # Величины смещений (biases) всех карт признаков слоя подвыборки
        self.__biases_of_feature_maps = [ (random.random() - 0.5) * 2.0 \
                                          for ind in range(features_maps_number) ]
        # Карты градиентов для соответствующих карт признаков слоя подвыборки
        self.__gradients = [ numpy.zeros(self.__feature_map_size) \
                             for ind in range(features_maps_number)]
        # Идентификатор слоя (как правило, номер слоя в нейронной сети)
        self.__layer_id = layer_id
        # Вспомогательные матрицы для представления уменьшенных входных карт
        self.__reduced_input_maps = [ numpy.zeros(self.__feature_map_size) \
                                      for ind in range(features_maps_number) ]

    @property
    def feature_map_size(self):
        return self.__feature_map_size

    @property
    def feature_maps_number(self):
        return len(self.__feature_maps)

    @property
    def input_map_size(self):
        return (self.__feature_map_size[0] * self.__receptive_field_size[0],
                self.__feature_map_size[1] * self.__receptive_field_size[1])

    @property
    def input_maps_number(self):
        return len(self.__feature_maps)

    @property
    def receptive_field_size(self):
        return self.__receptive_field_size

    @property
    def layer_id(self):
        return self.__layer_id

    @property
    def number_of_trainable_params(self):
        return len(self.__feature_maps) * 2

    @property
    def weights(self):
        return self.__weights_of_feature_maps

    @weights.setter
    def weights(self, values):
        err_msg = 'New values are not weights of {0} subsampling layer!'.format(
            cnn.common.integer_to_ordinal(self.__layer_id)
        )
        if len(values) != len(self.__feature_maps):
            raise TypeError(err_msg)
        for cur_value in values:
            if not isinstance(cur_value, float):
                raise TypeError(err_msg)
        self.__weights_of_feature_maps = copy.deepcopy(values)

    @property
    def biases(self):
        return self.__biases_of_feature_maps

    @biases.setter
    def biases(self, values):
        err_msg = 'New values are not biases of {0} subsampling layer!'.format(
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
        """ Инициализировать все веса (аддитивные смещения) и смещения случайными значениями. """
        features_maps_number = len(self.__feature_maps)
        self.__weights_of_feature_maps = [ (random.random() - 0.5) * 2.0 \
                                           for ind in range(features_maps_number) ]
        self.__biases_of_feature_maps = [ (random.random() - 0.5) * 2.0 \
                                          for ind in range(features_maps_number) ]

    def calculate_outputs(self, input_maps):
        """ Вычислить выходные карты слоя подвыборки для заданных входных данных.

        В ответ на подачу на вход слоя подвыборки входных карт input_maps (т.е. карт признаков
        предыдущего слоя свёртки) вычислить выходы карт признаков. Перед вычислением проверить
        допустимость структуры входных карт, которая должна соответствовать структуре слоя
        подвыборки.
        """
        self.__check_input_maps_of_this_layer(input_maps)
        for ft_ind in range(len(self.__feature_maps)):
            for ind in numpy.ndindex(self.__feature_map_size):
                row1 = ind[0] * self.__receptive_field_size[0]
                col1 = ind[1] * self.__receptive_field_size[1]
                row2 = row1 + self.__receptive_field_size[0]
                col2 = col1 + self.__receptive_field_size[1]
                self.__reduced_input_maps[ft_ind][ind] = input_maps[ft_ind][row1:row2,\
                                                                            col1:col2].max()
            self.__feature_maps[ft_ind] = numpy.tanh(
                self.__reduced_input_maps[ft_ind] * self.__weights_of_feature_maps[ft_ind] \
                + self.__biases_of_feature_maps[ft_ind]
            )
        return self.__feature_maps

    def calculate_gradient(self, next_weights, next_gradient, next_receptive_field_size):
        """ Вычислить карты градиентов слоя подвыборки.

        Вычислить карты градиентов слоя подвыборки, т.е. локальные градиенты нейронов всех карт
        признаков, на основе информации о весах нейронов next_weights, картах градиентов
        next_gradient и размере рецептивного поля следующего за ним слоя свёртки. Перед
        вычислением проверить соответствие параметров следующего слоя свёртки и текущего слоя
        подвыборки. При этом предполагается, что карты признаков слоя подвыборки уже были расчитаны
        на основе некоторых входных карт, поданных на вход этого слоя подвыборки (т.е. прямое
        распространение сигнала уже было, а сейчас идёт обратное распространение ошибки).
        """
        self.__check_gradient_maps_of_next_layer(next_weights, next_gradient, \
                                                 next_receptive_field_size)
        feature_maps_number = len(self.__feature_maps)
        next_feature_maps_number = len(next_gradient)
        next_feature_map_size = next_gradient[0].shape
        rotated_weights = [
            [numpy.flipud(numpy.fliplr(next_weights[next_ft_ind][ft_ind])) \
             for ft_ind in range(feature_maps_number)] \
            for next_ft_ind in range(next_feature_maps_number)
        ]
        tmp_matrix = numpy.zeros((next_feature_map_size[0] + 2 * next_receptive_field_size[0] - 2,
                                  next_feature_map_size[1] + 2 * next_receptive_field_size[1] - 2))
        row1 = next_receptive_field_size[0] - 1
        col1 = next_receptive_field_size[1] - 1
        row2 = row1 + next_feature_map_size[0]
        col2 = col1 + next_feature_map_size[1]
        for ft_ind in range(feature_maps_number):
            self.__gradients[ft_ind] = numpy.zeros(self.__feature_map_size)
            for next_ft_ind in range(next_feature_maps_number):
                tmp_matrix[row1:row2, col1:col2] = next_gradient[next_ft_ind]
                for ind in numpy.ndindex(self.__feature_map_size):
                    row_slice = (ind[0], (ind[0] + next_receptive_field_size[0]))
                    col_slice = (ind[1], (ind[1] + next_receptive_field_size[1]))
                    self.__gradients[ft_ind][ind] += numpy.sum(
                        tmp_matrix[row_slice[0]:row_slice[1], col_slice[0]:col_slice[1]] \
                        * rotated_weights[next_ft_ind][ft_ind]
                    )
            self.__gradients[ft_ind] *= (1.0 - self.__feature_maps[ft_ind]\
                                         * self.__feature_maps[ft_ind])
        return self.__gradients

    def update_weights_and_biases(self, learning_rate):
        """ Обновить веса и смещения всех нейронов слоя подвыборки.

        После того, как в результате прямого распространения сигнала были расчитаны карты признаков,
        а в результате обратного распространения ошибки - карты градиентов, обновить веса
        (аддитивные смещения) и смещения нейронов слоя подвыборки. """
        if learning_rate is None:
            return
        for ft_ind in range(len(self.__feature_maps)):
            delta_w = numpy.sum(self.__gradients[ft_ind] * self.__reduced_input_maps[ft_ind])
            self.__weights_of_feature_maps[ft_ind] += delta_w * learning_rate
            delta_b = numpy.sum(self.__gradients[ft_ind])
            self.__biases_of_feature_maps[ft_ind] += delta_b * learning_rate

    def __check_input_maps_of_this_layer(self, input_maps):
        """ Проверить корректность структуры входных карт input_maps.

        Сгенерировать исключение ESubsamplingLayerCalculating в ситуации, если заданные входные
        карты не являются массивами numpy.ndarray, или же их количество или их размеры не
        соответствуют ожидаемым.
        """
        if input_maps is None:
            raise ESubsamplingLayerCalculating(self.__layer_id, 'Input maps are undefined!')
        number_of_input_maps = len(input_maps)
        if number_of_input_maps != len(self.__feature_maps):
            raise ESubsamplingLayerCalculating(self.__layer_id,
                                               'Number of input maps is incorrect!')
        if not isinstance(input_maps[0], numpy.ndarray):
            raise ESubsamplingLayerCalculating(self.__layer_id,
                                               'Type of 1st input map is inadmissible!')
        input_size = input_maps[0].shape
        if len(input_size) != 2:
            raise ESubsamplingLayerCalculating(self.__layer_id,
                                               '1st input map is not two-dimensional!')
        if input_size[0] != (self.__feature_map_size[0] * self.__receptive_field_size[0]):
            raise ESubsamplingLayerCalculating(self.__layer_id,
                                               'Row number of 1st input map is incorrect!')
        if input_size[1] != (self.__feature_map_size[1] * self.__receptive_field_size[1]):
            raise ESubsamplingLayerCalculating(self.__layer_id,
                                               'Column number of 1st input map is incorrect!')
        for ind in range(number_of_input_maps-1):
            if not isinstance(input_maps[ind+1], numpy.ndarray):
                raise ESubsamplingLayerCalculating(
                    self.__layer_id, 'Type of {0} input map is inadmissible!'.format(
                        cnn.common.integer_to_ordinal(ind+2))
                )
            if input_maps[ind+1].shape != input_maps[ind].shape:
                raise ESubsamplingLayerCalculating(
                    self.__layer_id,
                    'Size of {0} input map does not coinside with size of other '\
                    'input maps!'.format(cnn.common.integer_to_ordinal(ind+2))
                )

    def __check_gradient_maps_of_next_layer(self, next_weights, next_gradient_maps, \
                                            next_receptive_field_size):
        """ Сгенерировать исключение ESubsamplingLayerGradient, если параметры следующего слоя
        свёртки заданы некорректно по отношению к текущему слою подвыборки.
        """
        if next_weights is None:
            raise ESubsamplingLayerGradient(
                self.__layer_id, 'Weights of neuron in next layer are undefined!'
            )
        if next_gradient_maps is None:
            raise ESubsamplingLayerGradient(
                self.__layer_id, 'Gradient maps in next layer are undefined!'
            )
        if next_receptive_field_size is None:
            raise ESubsamplingLayerGradient(
                self.__layer_id, 'Receptive field size of next layer is undefined!'
            )
        number_of_next_maps = len(next_gradient_maps)
        if number_of_next_maps == 0:
            raise ESubsamplingLayerGradient(
                self.__layer_id, 'Gradient maps in next layer are undefined!'
            )
        if not isinstance(next_gradient_maps[0], numpy.ndarray):
            raise ESubsamplingLayerGradient(
                self.__layer_id, 'Type of 1st gradient map in next layer is inadmissible!'
            )
        next_layer_shape = next_gradient_maps[0].shape
        if len(next_layer_shape) != 2:
            raise ESubsamplingLayerGradient(
                self.__layer_id, '1st gradient map in next layer is not two-dimensional!'
            )
        if len(next_receptive_field_size) != 2:
            raise ESubsamplingLayerGradient(
                self.__layer_id, 'Receptive field of next layer is not two-dimensional!'
            )
        if (self.__feature_map_size[0] - (next_receptive_field_size[0] - 1)) != next_layer_shape[0]:
            raise ESubsamplingLayerGradient(
                self.__layer_id, 'Row number of first gradient map in next layer does not '\
                'correspond to row number of feature maps in this layer.'
            )
        if (self.__feature_map_size[1] - (next_receptive_field_size[1] - 1)) != next_layer_shape[1]:
            raise ESubsamplingLayerGradient(
                self.__layer_id, 'Column number of first gradient map in next layer does not '\
                'correspond to column number of feature maps in this layer.'
            )
        for ind in range(number_of_next_maps-1):
            if not isinstance(next_gradient_maps[ind+1], numpy.ndarray):
                raise ESubsamplingLayerGradient(
                    self.__layer_id,
                    'Type of {0} gradient map in next layer is '\
                    'inadmissible!'.format(cnn.common.integer_to_ordinal(ind+2))
                )
            if next_gradient_maps[ind+1].shape != next_layer_shape:
                raise ESubsamplingLayerGradient(
                    self.__layer_id,
                    'Size of {0} gradient map in next layer does not coinside with size of other '\
                    'gradient maps in that layer!'.format(cnn.common.integer_to_ordinal(ind+2))
                )
        if len(next_weights) != number_of_next_maps:
            raise ESubsamplingLayerGradient(
                self.__layer_id, 'Structure of weights (convolution kernels) in next (convolution)'\
                ' layer does not correspond to feature maps number in this layer.'
            )
        for ft_ind in range(number_of_next_maps):
            if len(next_weights[ft_ind]) != len(self.__feature_maps):
                raise ESubsamplingLayerGradient(
                    self.__layer_id, 'Convolution kernels number in {0} feature map of next layer '\
                    'does not correspond to feature maps number in this '\
                    'layer.'.format(cnn.common.integer_to_ordinal(ft_ind+1))
                )
            for inp_ind in range(len(next_weights[ft_ind])):
                if not isinstance(next_weights[ft_ind][inp_ind], numpy.ndarray):
                    raise ESubsamplingLayerGradient(
                        self.__layer_id,
                        'Type of {0} convolution kernel in {1} feature map of next (convolution) '\
                        'layer is inadmissible!'.format(cnn.common.integer_to_ordinal(inp_ind+1),
                                                        cnn.common.integer_to_ordinal(ft_ind+1))
                    )
                if next_weights[ft_ind][inp_ind].shape != next_receptive_field_size:
                    raise ESubsamplingLayerGradient(
                        self.__layer_id,
                        'Sizes of {0} convolution kernel in {1} feature map of next (convolution) '\
                        'layer are incorrect!'.format(cnn.common.integer_to_ordinal(inp_ind+1),
                                                      cnn.common.integer_to_ordinal(ft_ind+1))
                    )
