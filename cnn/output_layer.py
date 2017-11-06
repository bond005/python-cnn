# -*- coding: utf-8 -*-
#
# Скрипт ориентирован на использование в Python 3.*
#

import copy
import functools
import math
import numpy
import random

import cnn.common


class EOutputLayerCreating(cnn.common.ELayerException):
    """ Ошибка генерируется внутри конструктора класса OutputLayer в случае, если параметры
    создаваемого выходного слоя заданы неверно. """
    def get_base_msg(self):
        base_msg = 'output layer cannot be created!'
        layer_str = self.get_layer_id_as_string()
        if len(layer_str) > 0:
            base_msg = '{0} (output) layer cannot be created!'.format(layer_str)
        return base_msg

class EOutputLayerCalculating(cnn.common.ELayerException):
    """ Ошибка генерируется в случае, если данные (входные карты), поданные на вход выходного слоя,
    некорректны, т.е. не соответствуют по своей структуре этому слою. """
    def get_base_msg(self):
        base_msg = 'Outputs of output layer cannot be calculated!'
        layer_str = self.get_layer_id_as_string()
        if len(layer_str) > 0:
            base_msg = 'Outputs of {0} (output) layer cannot be '\
                       'calculated!'.format(layer_str)
        return base_msg

class EOutputLayerGradient(cnn.common.ELayerException):
    """ Ошибка генерируется в случае, если невозможно расчитать карты градиентов для выходного слоя
    на основе заданного вектора желаемых выходных сигналов из обучающего множества (вектор желаемых
    выходных сигналов не соответствует структуре выходного слоя). """
    def get_base_msg(self):
        base_msg = 'Gradient of output layer cannot be calculated!'
        layer_str = self.get_layer_id_as_string()
        if len(layer_str) > 0:
            base_msg = 'Gradient of {0} (output) layer cannot be '\
                       'calculated!'.format(layer_str)
        return base_msg


class OutputLayer:
    """ Выходной слой.

    Вычисления в данном слое реализованы так же, как и в слое свёртки, со следующими нюансами:

    1) каждая фича-мапа выходного слоя имеет размер 1x1, т.е. по сути представляет собой один
    нейрон;
    2) размер рецептивного поля выходного слоя равен размеру входной фича-мапы;
    3) активационная функция нейронов выходного слоя - функция SOFTMAX;
    4) поскольку для нейронов выходного слоя, в отличии от нейронов обычного слоя свёртки,
    известны желаемые выходные сигналы, то локальные градиенты нейронов выходного слоя вычисляются
    непосредственно, а не через локальные градиенты нейронов следующего слоя;
    5) целевая функция при обучении нейронной сети - кросс-энтропия.
    """
    def __init__(self, layer_id, input_maps_number, input_map_size, neurons_number):
        if layer_id is None:
            raise EOutputLayerCreating('Integer identifier of this layer is specified incorrectly!')
        if not isinstance(layer_id, int):
            raise EOutputLayerCreating('Integer identifier of this layer is specified incorrectly!')
        if layer_id < 1:
            raise EOutputLayerCreating('Integer identifier of this layer is specified incorrectly!')
        if (input_maps_number < 1) or (len(input_map_size) != 2):
            raise EOutputLayerCreating(layer_id,
                                       'Structure of input data for this layer (number of input '\
                                       'maps) is specified incorrectly!')
        if (input_map_size[0] < 1) or (input_map_size[1] < 1):
            raise EOutputLayerCreating(layer_id,
                                       'Structure of input data for this layer (number of input '\
                                       'maps) is specified incorrectly!')
        if neurons_number < 2:
            raise EOutputLayerCreating(layer_id, 'Number of neurons in this layer is specified '\
                                       'incorrectly!')
        # Количество входных карт в данных, подаваемых на вход слоя свёртки
        self.__number_of_input_maps = input_maps_number
        # Размер одной входной карты в данных, подаваемых на вход слоя свёртки
        self.__input_map_size = input_map_size
        # Список выходных сигналов нейронов выходного слоя
        self.__outputs = [0.0 for ind in range(neurons_number)]
        # Веса всех нейронов выходного слоя
        self.__weights_of_neurons = list()
        for neuron_ind in range(neurons_number):
            self.__weights_of_neurons.append(
                [ (numpy.random.random(self.__input_map_size) - 0.5) * 2.0 \
                  for inp_map_ind in range(input_maps_number) ]
            )
        # Величины смещений (biases) всех нейронов выходного слоя
        self.__biases_of_neurons = [ (random.random() - 0.5) * 2.0 \
                                     for ind in range(neurons_number) ]
        # Градиенты для соответствующих нейронов выходного слоя
        self.__gradients = [0.0 for ind in range(neurons_number)]
        # Идентификатор слоя (как правило, номер слоя в нейронной сети)
        self.__layer_id = layer_id

    @property
    def neurons_number(self):
        return len(self.__outputs)

    @property
    def input_map_size(self):
        return self.__input_map_size

    @property
    def input_maps_number(self):
        return self.__number_of_input_maps

    @property
    def layer_id(self):
        return self.__layer_id

    @property
    def number_of_trainable_params(self):
        return (self.__input_map_size[0] * self.__input_map_size[1] \
                * self.__number_of_input_maps + 1) * len(self.__outputs)

    @property
    def weights(self):
        return self.__weights_of_neurons

    @weights.setter
    def weights(self, values):
        err_msg = 'New values are not weights of {0} (output) layer!'.format(
            cnn.common.integer_to_ordinal(self.__layer_id)
        )
        if len(values) != len(self.__outputs):
            raise TypeError(err_msg)
        for weights_of_neuron in values:
            if len(weights_of_neuron) != self.__number_of_input_maps:
                raise TypeError(err_msg)
            for conv_cernel in weights_of_neuron:
                if not isinstance(conv_cernel, numpy.ndarray):
                    raise TypeError(err_msg)
                if conv_cernel.dtype.name != 'float64':
                    raise TypeError(err_msg)
                if conv_cernel.shape != self.__input_map_size:
                    raise TypeError(err_msg)
        self.__weights_of_neurons = copy.deepcopy(values)

    @property
    def biases(self):
        return self.__biases_of_neurons

    @biases.setter
    def biases(self, values):
        err_msg = 'New values are not biases of {0} (output) layer!'.format(
            cnn.common.integer_to_ordinal(self.__layer_id)
        )
        if len(values) != len(self.__outputs):
            raise TypeError(err_msg)
        for cur_value in values:
            if not isinstance(cur_value, float):
                raise TypeError(err_msg)
        self.__biases_of_neurons = copy.deepcopy(values)

    @property
    def gradients(self):
        return self.__gradients

    def initialize_weights_and_biases(self):
        """ Инициализировать все веса и смещения случайными значениями. """
        neurons_number = len(self.__outputs)
        for neuron_ind in range(neurons_number):
            self.__weights_of_neurons[neuron_ind] = [
                (numpy.random.random(self.__input_map_size) - 0.5) * 2.0 \
                for inp_map_ind in range(self.__number_of_input_maps)
            ]
        self.__biases_of_neurons = [ (random.random() - 0.5) * 2.0 \
                                     for ind in range(neurons_number) ]

    def calculate_outputs(self, input_maps):
        """ Вычислить выходы нейронов выходного слоя для заданных входных данных.

        В ответ на подачу на вход выходного слоя входных карт input_maps вычислить выходы нейронов.
        Перед вычислением проверить допустимость структуры входных карт, которая должна
        соответствовать структуре выходного слоя.
        """
        self.__check_input_maps_of_this_layer(input_maps)
        sum_of_outputs = 0.0
        for neuron_ind in range(len(self.__outputs)):
            output_of_accumulator = functools.reduce(
                lambda a, b: a + numpy.sum(b[0] * b[1]),
                zip(input_maps, self.__weights_of_neurons[neuron_ind]),
                self.__biases_of_neurons[neuron_ind]
            )
            self.__outputs[neuron_ind] = math.exp(output_of_accumulator)
            sum_of_outputs += self.__outputs[neuron_ind]
        self.__outputs = list(map(lambda a: a / sum_of_outputs, self.__outputs))
        return self.__outputs

    def calculate_gradient(self, target_outputs):
        """ Вычислить градиенты нейронов выходного слоя.

        Вычислить локальные градиенты всех нейронов выходного слоя на основе информации о векторе
        желаемых выходных сигналов target_outputs. Перед вычислением проверить соответствие
        структуры этого вектора структуре выходного слоя. При этом предполагается, что выходы
        нейронов выходного слоя уже были расчитаны на основе некоторых входных карт, поданных на
        вход этого слоя (т.е. прямое распространение сигнала уже было, а сейчас идёт обратное
        распространение ошибки).
        """
        self.__check_target_outputs(target_outputs)
        self.__gradients = [target_outputs[neuron_ind] - self.__outputs[neuron_ind]
                            for neuron_ind in range(len(self.__outputs))]
        return self.__gradients

    def update_weights_and_biases(self, learning_rate, input_maps):
        """ Обновить веса и смещения всех нейронов выходного слоя.

        После того, как в результате прямого распространения сигнала были расчитаны выходы нейронов,
        а в результате обратного распространения ошибки - локальные градиенты этих нейронов,
        обновить веса и смещения нейронов выходного слоя. """
        self.__check_input_maps_of_this_layer(input_maps)
        if learning_rate is None:
            return
        self.__biases_of_neurons = [ self.__biases_of_neurons[neuron_ind]
                                     + self.__gradients[neuron_ind] * learning_rate
                                     for neuron_ind in range(len(self.__outputs)) ]
        for neuron_ind in range(len(self.__outputs)):
            self.__weights_of_neurons[neuron_ind] = list(map(
                lambda a, b: a * self.__gradients[neuron_ind] * learning_rate + b,
                input_maps, self.__weights_of_neurons[neuron_ind]
            ))

    def __check_input_maps_of_this_layer(self, input_maps):
        """ Проверить корректность структуры входных карт input_maps.

        Сгенерировать исключение EOutputLayerCalculating в ситуации, если заданные входные
        карты не являются массивами numpy.ndarray, или же их количество или их размеры не
        соответствуют ожидаемым.
        """
        if input_maps is None:
            raise EOutputLayerCalculating(self.__layer_id, 'Input maps are undefined!')
        number_of_input_maps = len(input_maps)
        if number_of_input_maps != self.__number_of_input_maps:
            raise EOutputLayerCalculating(self.__layer_id, 'Number of input maps is incorrect!')
        if not isinstance(input_maps[0], numpy.ndarray):
            raise EOutputLayerCalculating(self.__layer_id, 'Type of 1st input map is inadmissible!')
        input_size = input_maps[0].shape
        if len(input_size) != 2:
            raise EOutputLayerCalculating(self.__layer_id, '1st input map is not two-dimensional!')
        if input_size[0] != self.__input_map_size[0]:
            raise EOutputLayerCalculating(self.__layer_id,
                                          'Row number of 1st input map is incorrect!')
        if input_size[1] != self.__input_map_size[1]:
            raise EOutputLayerCalculating(self.__layer_id,
                                          'Column number of 1st input map is incorrect!')
        for ind in range(number_of_input_maps-1):
            if not isinstance(input_maps[ind+1], numpy.ndarray):
                raise EOutputLayerCalculating(
                    self.__layer_id, 'Type of {0} input map is inadmissible!'.format(
                        cnn.common.integer_to_ordinal(ind+2))
                )
            if input_maps[ind+1].shape != input_maps[ind].shape:
                raise EOutputLayerCalculating(
                    self.__layer_id,
                    'Size of {0} input map does not coinside with size of other '\
                    'input maps!'.format(cnn.common.integer_to_ordinal(ind+2))
                )

    def __check_target_outputs(self, target_outputs):
        """ Сгенерировать исключение EOutputLayerGradient, если структура желаемого выходного
        сигнала не соответствует структуре выходного слоя.
        """
        if target_outputs is None:
            raise EOutputLayerGradient(
                self.__layer_id, 'Target outputs are undefined!'
            )
        if len(self.__outputs) > 10000:
            epsilon = 0.00000001
        else:
            epsilon = 0.0001 / float(len(self.__outputs))
        if isinstance(target_outputs, numpy.ndarray):
            target_size = target_outputs.shape
            if len(target_size) != 1:
                raise EOutputLayerGradient(
                    self.__layer_id, 'Structure of target outputs is incorrect!'
                )
            if target_size[0] != len(self.__outputs):
                raise EOutputLayerGradient(
                    self.__layer_id, 'Number of target outputs is incorrect!'
                )
            outputs_sum = 0.0
            for ind in range(len(self.__outputs)):
                if (target_outputs[ind] < 0.0) or (target_outputs[ind] > 1.0):
                    raise EOutputLayerGradient(
                        self.__layer_id,
                        'Value of {0} target output is incorrect!'.format(
                            cnn.common.integer_to_ordinal(ind+1)
                        )
                    )
                outputs_sum += target_outputs[ind]
            if math.fabs(outputs_sum - 1.0) > epsilon:
                raise EOutputLayerGradient(
                    self.__layer_id,
                    'Target outputs are incorrect! Sum of target outputs does not '\
                    'equal to 1.'.format(cnn.common.integer_to_ordinal(ind+1))
                )
        else:
            if len(target_outputs) != len(self.__outputs):
                raise EOutputLayerGradient(
                    self.__layer_id, 'Number of target outputs is incorrect!'
                )
            outputs_sum = 0.0
            for ind in range(len(self.__outputs)):
                if not isinstance(target_outputs[ind], float):
                    raise EOutputLayerGradient(
                        self.__layer_id,
                        'Type of {0} target output is inadmissible!'.format(
                            cnn.common.integer_to_ordinal(ind+1)
                        )
                    )
                if (target_outputs[ind] < 0.0) or (target_outputs[ind] > 1.0):
                    raise EOutputLayerGradient(
                        self.__layer_id,
                        'Value of {0} target output is incorrect!'.format(
                            cnn.common.integer_to_ordinal(ind+1)
                        )
                    )
                outputs_sum += target_outputs[ind]
            if math.fabs(outputs_sum - 1.0) > epsilon:
                raise EOutputLayerGradient(
                    self.__layer_id,
                    'Target outputs are incorrect! Sum of target outputs does not '\
                    'equal to 1.'.format(cnn.common.integer_to_ordinal(ind+1))
                )
