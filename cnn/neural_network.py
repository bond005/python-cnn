from functools import reduce
import pickle
import random
import tempfile

import numpy
from scipy.sparse import csc_matrix

from cnn.common import ENeuralNetworkException, integer_to_ordinal
from cnn.convolution_layer import ConvolutionLayer
from cnn.subsampling_layer import SubsamplingLayer
from cnn.output_layer import OutputLayer

class ECNNCreating(ENeuralNetworkException):
    """ Ошибка создания свёрточной НС. """
    def get_base_msg(self):
        return 'Convolution neural network cannot be created!'


class ECNNTraining(ENeuralNetworkException):
    """ Ошибка обучения свёрточной НС. """


class CNNClassifier:
    def __init__(self, input_maps_number, input_map_size, structure_of_hidden_layers, classes_number,
                 max_epochs_number=10, learning_rate=0.0001, early_stopping=True, validation_fraction=0.1):
        if max_epochs_number < 1:
            raise ECNNCreating('Maximal number of training epochs is specified incorrectly!')
        if learning_rate <= 0.0:
            raise ECNNCreating('Learning rate parameter is specified incorrectly!')
        if early_stopping:
            if (validation_fraction <= 0.0) or (validation_fraction >= 1.0):
                raise ECNNCreating('Validation fraction is specified incorrectly!')
        if (input_maps_number < 1) or (len(input_map_size) != 2):
            raise ECNNCreating('Input maps for CNN are specified incorrectly!')
        if (input_map_size[0] < 1) or (input_map_size[1] < 1):
            raise ECNNCreating('Input maps for CNN are specified incorrectly!')
        if classes_number < 2:
            raise ECNNCreating('Number of classes is too small!')
        number_of_hidden_layers = len(structure_of_hidden_layers)
        if number_of_hidden_layers < 2:
            raise ECNNCreating('Number of hidden layers is too small!')
        if number_of_hidden_layers % 2 != 0:
            raise ECNNCreating('Number of hidden layers is incorrect (it must be even value)!')
        for hidden_layer_ind in range(number_of_hidden_layers):
            hidden_layer_structure = structure_of_hidden_layers[hidden_layer_ind]
            if not isinstance(hidden_layer_structure, dict):
                raise ECNNCreating('Structure of layer {0} is specified incorrectly!'.format(hidden_layer_ind + 1))
            if 'feature_maps_number' not in hidden_layer_structure:
                raise ECNNCreating('Structure of layer {0} is specified incorrectly!'.format(hidden_layer_ind + 1))
            if 'feature_map_size' not in hidden_layer_structure:
                raise ECNNCreating('Structure of layer {0} is specified incorrectly!'.format(hidden_layer_ind + 1))
            if 'receptive_field_size' not in hidden_layer_structure:
                raise ECNNCreating('Structure of layer {0} is specified incorrectly!'.format(hidden_layer_ind + 1))
        self.__input_maps_number = input_maps_number
        self.__input_map_size = input_map_size
        self.__layers = list()
        self.__layers.append(
            ConvolutionLayer(1, input_maps_number, structure_of_hidden_layers[0]['feature_maps_number'],
                             structure_of_hidden_layers[0]['feature_map_size'],
                             structure_of_hidden_layers[0]['receptive_field_size'])
        )
        if tuple(self.__layers[0].input_map_size) != tuple(input_map_size):
            raise ECNNCreating('Structure of layer 1 does not correspond to structure of input maps!')
        for ind in range(number_of_hidden_layers / 2 - 1):
            hidden_layer_ind = ind * 2 + 1
            self.__layers.append(
                SubsamplingLayer(hidden_layer_ind + 1,
                                 structure_of_hidden_layers[hidden_layer_ind]['feature_maps_number'],
                                 structure_of_hidden_layers[hidden_layer_ind]['feature_map_size'],
                                 structure_of_hidden_layers[hidden_layer_ind]['receptive_field_size'])
            )
            if self.__layers[hidden_layer_ind].input_map_size != self.__layers[hidden_layer_ind-1].feature_map_size:
                raise ECNNCreating('Structure of layer {0} does not correspond to structure of previous layer!'.format(
                    hidden_layer_ind + 1)
                )
            hidden_layer_ind += 1
            self.__layers.append(
                ConvolutionLayer(hidden_layer_ind + 1,
                                 structure_of_hidden_layers[hidden_layer_ind-1]['feature_maps_number'],
                                 structure_of_hidden_layers[hidden_layer_ind]['feature_maps_number'],
                                 structure_of_hidden_layers[hidden_layer_ind]['feature_map_size'],
                                 structure_of_hidden_layers[hidden_layer_ind]['receptive_field_size'])
            )
            if self.__layers[hidden_layer_ind].input_map_size != self.__layers[hidden_layer_ind - 1].feature_map_size:
                raise ECNNCreating('Structure of layer {0} does not correspond to structure of previous layer!'.format(
                    hidden_layer_ind + 1)
                )
        hidden_layer_ind = number_of_hidden_layers - 2
        self.__layers.append(
            SubsamplingLayer(hidden_layer_ind + 1, structure_of_hidden_layers[hidden_layer_ind]['feature_maps_number'],
                             structure_of_hidden_layers[hidden_layer_ind]['feature_map_size'],
                             structure_of_hidden_layers[hidden_layer_ind]['receptive_field_size'])
        )
        if self.__layers[hidden_layer_ind].input_map_size != self.__layers[hidden_layer_ind - 1].feature_map_size:
            raise ECNNCreating('Structure of layer {0} does not correspond to structure of previous layer!'.format(
                hidden_layer_ind + 1)
            )
        hidden_layer_ind += 1
        self.__layers.append(
            OutputLayer(hidden_layer_ind, structure_of_hidden_layers[hidden_layer_ind-1]['feature_maps_number'],
                        structure_of_hidden_layers[hidden_layer_ind - 1]['feature_map_size'],
                        classes_number if classes_number > 2 else 1)
        )
        self.__max_epochs_number = max_epochs_number
        self.__learning_rate = learning_rate
        self.__early_stopping = early_stopping
        self.__validation_fraction = validation_fraction

    @property
    def input_maps_number(self):
        return self.__input_maps_number

    @property
    def input_map_size(self):
        return self.__input_map_size

    @property
    def layers(self):
        return self.__layers

    @property
    def max_train_epochs(self):
        return self.__max_epochs_number

    @max_train_epochs.setter
    def max_train_epochs(self, new_value):
        if (not isinstance(new_value, int)) or (new_value < 1):
            raise TypeError('Maximal number of training epochs is specified incorrectly!')
        self.__max_epochs_number = new_value

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, new_value):
        if (not isinstance(new_value, float)) or (new_value <= 0.0):
            raise TypeError('Learning rate parameter is specified incorrectly!')
        self.__learning_rate = new_value

    @property
    def early_stopping(self):
        return self.__early_stopping

    @early_stopping.setter
    def early_stopping(self, new_value):
        if not isinstance(new_value, bool):
            raise TypeError('Usage of early stopping is specified incorrectly!')
        self.__early_stopping = new_value

    @property
    def validation_fraction(self):
        return self.__validation_fraction

    @validation_fraction.setter
    def validation_fraction(self, new_value):
        if (not isinstance(new_value, float)) or (new_value <= 0.0) or (new_value >= 1.0):
            raise TypeError('Validation fraction is specified incorrectly!')
        self.__validation_fraction = new_value

    def fit(self, X, y):
        err_msg = self.__check_input_data(X)
        if len(err_msg) > 0:
            raise ECNNTraining(err_msg)
        err_msg = self.__check_target_data(X.shape[0], y)
        if len(err_msg) > 0:
            raise ECNNTraining(err_msg)
        if self.__early_stopping:
            size_of_validation_set = int(round(X.shape[0] * self.__validation_fraction))
            if (size_of_validation_set < 1) or (size_of_validation_set >= X.shape[0]):
                raise ECNNTraining('Validation set is specified incorrectly!')
            indexes_of_samples = range(X.shape[0])
            random.shuffle(indexes_of_samples)
            indexes_of_train_samples = indexes_of_samples[:size_of_validation_set]
            indexes_of_validation_samples = indexes_of_samples[size_of_validation_set:]
            repeats = 10
            while repeats > 0:
                set_of_train_labels = set()
                for ind in indexes_of_train_samples:
                    set_of_train_labels.add(y[ind])
                set_of_validation_labels = set()
                for ind in indexes_of_validation_samples:
                    set_of_validation_labels.add(y[ind])
                if set_of_train_labels == set_of_validation_labels:
                    break
                random.shuffle(indexes_of_samples)
                indexes_of_train_samples = indexes_of_samples[:size_of_validation_set]
                indexes_of_validation_samples = indexes_of_samples[size_of_validation_set:]
                repeats -= 1
            if repeats < 1:
                raise ECNNTraining('Input data cannot be divided into train subset and validation subset.')
            X_valid = X[indexes_of_validation_samples]
            y_valid = y[indexes_of_validation_samples]
            X_train = X[indexes_of_train_samples]
            y_train = csc_matrix((1.0, (numpy.arange(len(indexes_of_train_samples)), y[indexes_of_train_samples])),
                                 shape=(len(indexes_of_train_samples), self.__layers[-1].neurons_number))
        else:
            X_train = X
            y_train = csc_matrix((1.0, (numpy.arange(X.shape[0]), y)),
                                 shape=(X.shape[0], self.__layers[-1].neurons_number))
            X_valid = None
            y_valid = None
        indexes_of_train_samples = range(X_train.shape[0])
        if (X_valid is not None) and (y_valid is not None):
            epochs_number = 0
            can_continue = True
            previous_validation_error = self.score(X_valid, y_valid)
            counter_for_early_stopping = 0
            with tempfile.TemporaryFile() as tmp_cnn_fp:
                pickle.dump(tmp_cnn_fp, self.__layers)
                while can_continue:
                    random.shuffle(indexes_of_train_samples)
                    self.__do_train_epoch(X_train, y_train, indexes_of_train_samples)
                    current_validation_error = self.score(X_valid, y_valid)
                    if current_validation_error < previous_validation_error:
                        counter_for_early_stopping = 0
                        tmp_cnn_fp.truncate(0)
                        pickle.dump(tmp_cnn_fp, self.__layers)
                    else:
                        counter_for_early_stopping += 1
                        if counter_for_early_stopping >= 3:
                            can_continue = False
                    epochs_number += 1
                    if can_continue and (epochs_number >= self.__max_epochs_number):
                        can_continue = False
                tmp_cnn_fp.seek(0)
                self.__layers = pickle.load(tmp_cnn_fp)
        else:
            for epoch in range(self.__max_epochs_number):
                random.shuffle(indexes_of_train_samples)
                self.__do_train_epoch(X_train, y_train, indexes_of_train_samples)
            epochs_number = self.__max_epochs_number
        return epochs_number

    def predict(self, X):
        y = numpy.empty(shape=(X.shape[0],), dtype=numpy.uint32)
        for ind in range(X.shape[0]):
            y[ind] = self.__calculate_output_label(X[ind])
        return y

    def score(self, X, y):
        return reduce(
            lambda a, b: a + b,
            map(lambda y_true, x: 1.0 if y_true == self.__calculate_output_label(x) else 0.0, y, X),
            0.0
        )

    def __calculate_output_label(self, X):
        outputs_of_layer = self.__layers[0].calculate_outputs(X)
        for layer_ind in range(len(self.__layers) - 1):
            outputs_of_layer = self.__layers[layer_ind+1].calculate_outputs(outputs_of_layer)
        return max(enumerate(outputs_of_layer), key=lambda x: x[1])[0]

    def __check_input_data(self, X):
        err_msg = ''
        base_err_msg = 'Structure of input data does not correspond to structure of neural network.'
        if isinstance(X, numpy.ndarray):
            sizes = X.shape
            if len(sizes) == 3:
                if self.__input_maps_number > 1:
                    err_msg = 'Each input sample must consist of {0} input maps.'.format(self.__input_maps_number)
                elif (sizes[1] != self.__input_map_size[0]) or (sizes[2] != self.__input_map_size[1]):
                    err_msg = 'Each input map in input data samples must be {0}-by-{1} matrix.'.format(
                        self.__input_map_size[0], self.__input_map_size[1]
                    )
            elif len(sizes) == 4:
                if sizes[1] != self.__input_maps_number:
                    if self.__input_maps_number > 1:
                        err_msg = 'Each input sample must consist of {0} input maps.'.format(self.__input_maps_number)
                    else:
                        err_msg = 'Each input sample must consist of only one input map.'
                elif (sizes[2] != self.__input_map_size[0]) or (sizes[3] != self.__input_map_size[1]):
                    err_msg = 'Each input map in input data samples must be {0}-by-{1} matrix.'.format(
                        self.__input_map_size[0], self.__input_map_size[1]
                    )
            else:
                if len(sizes) > 4:
                    err_msg = 'Input data has too many dimensions.'
                else:
                    err_msg = 'Input data has too few dimensions.'
        else:
            err_msg = 'Input data must be numpy.ndarray.'
        if len(err_msg) > 0:
            err_msg = base_err_msg + ' ' + err_msg
        return err_msg

    def __check_target_data(self, number_of_samples, y):
        err_msg = ''
        if isinstance(y, numpy.ndarray):
            set_of_integer_types = {numpy.int32, numpy.uint32, numpy.int64, numpy.uint64, numpy.int16, numpy.uint16,
                                    numpy.int8, numpy.uint8, numpy.int_, numpy.intc, numpy.intp}
            if y.dtype in set_of_integer_types:
                sizes = y.shape
                if len(sizes) > 1:
                    err_msg = 'Target output has too many dimensions.'
                else:
                    if sizes[0] == number_of_samples:
                        for ind in range(number_of_samples):
                            class_index = int(y[ind])
                            if (class_index < 0) or (class_index >= self.__layers[-1].neurons_number):
                                err_msg = 'Target output for {0} sample is incorrect.'.format(
                                    integer_to_ordinal(ind + 1)
                                )
                                break
                    else:
                        err_msg = 'Number of target outputs does not correspond to number of input samples.'
            else:
                err_msg = 'Each value of target output must be integer.'
        else:
            err_msg = 'Target output must be numpy.ndarray.'
        return err_msg

    def __do_train_epoch(self, X, y, indexes):
        number_of_layers = len(self.__layers)
        for cur_ind in indexes:
            outputs_of_layer = self.__layers[0].calculate_outputs(X[cur_ind])
            for layer_ind in range(number_of_layers - 1):
                outputs_of_layer = self.__layers[layer_ind + 1].calculate_outputs(outputs_of_layer)
            self.__layers[-1].calculate_gradient(y[cur_ind].toarray())
            for layer_ind in range(number_of_layers - 1):
                self.__layers[number_of_layers - 2 - layer_ind].calculate_gradient(
                    self.__layers[number_of_layers - 2 - layer_ind + 1].weights,
                    self.__layers[number_of_layers - 2 - layer_ind + 1].gradients,
                    self.__layers[number_of_layers - 2 - layer_ind + 1].receptive_field_size
                )
            for layer_ind in range(number_of_layers):
                self.__layers[layer_ind].update_weights_and_biases(self.__learning_rate)


