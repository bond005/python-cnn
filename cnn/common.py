# -*- coding: utf-8 -*-
#
# Скрипт ориентирован на использование в Python 3.*
#

def integer_to_ordinal(integer_value):
    """ Превратить целое положительное число в строку с соответствующим порядковым числительным. """
    if not isinstance(integer_value, int):
        return ''
    if integer_value < 1:
        return ''
    ordinal_str = str(integer_value)
    if (len(ordinal_str) == 2) and (ordinal_str[-2] == '1'):
        ordinal_str += 'th'
    else:
        if ordinal_str[-1] == '1':
            ordinal_str += 'st'
        elif ordinal_str[-1] == '2':
            ordinal_str += 'nd'
        elif ordinal_str[-1] == '3':
            ordinal_str += 'rd'
        else:
            ordinal_str += 'th'
    return ordinal_str

class ELayerException(RuntimeError):
    """ Ошибка - предок всех ошибок, специфичных для слоёв свёрточной НС. """
    def __init__(self, layer_id, msg = None):
        Exception.__init__(self)
        self.__error_msg = ''
        self.__layer_id = 0
        if (layer_id is not None) and isinstance(layer_id, int):
            if layer_id >= 1:
                self.__layer_id = layer_id
        self.__error_msg = self.get_base_msg()
        if msg is not None:
            self.__error_msg += (' ' + msg.strip())

    def get_layer_id_as_string(self):
        return integer_to_ordinal(self.__layer_id)

    def get_base_msg(self):
        raise NotImplementedError()

    def __str__(self):
        return self.__error_msg
