import numpy

from cnn.convolution_layer import ConvolutionLayer
from cnn.subsampling_layer import SubsamplingLayer
from cnn.output_layer import OutputLayer

class ECNNCreating(RuntimeError):
    """ Ошибка создания свёрточной НС. """
    def __init__(self, msg=None):
        Exception.__init__(self)
        self.__error_msg = 'Convolution neural network cannot be created!'
        if msg is not None:
            self.__error_msg += (' ' + msg.strip())

    def __str__(self):
        return self.__error_msg


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
        if self.__early_stopping:
            size_of_validation_set = int(round(X.shape[0] * self.__validation_fraction))
            if (size_of_validation_set < 1) or (size_of_validation_set >= X.shape[0]):
                pass
        pass

    def predict(self, X):
        pass

    def score(self, X, y):
        pass

    def __check_input_data(self, X):
        if isinstance(X, numpy.ndarray):
            sizes = X.shape
            if len(sizes) == 3:
                if self.__input_maps_number > 1:
                    pass
                if (sizes[1] != self.__input_map_size[0]) or (sizes[2] != self.__input_map_size[1]):
                    pass
            elif len(sizes) == 4:
                if sizes[1] != self.__input_maps_number:
                    pass
                if (sizes[2] != self.__input_map_size[0]) or (sizes[3] != self.__input_map_size[1]):
                    pass
            else:
                pass
        elif (isinstance(X, list)) or (isinstance(X, tuple)):
            pass
        else:
            pass

