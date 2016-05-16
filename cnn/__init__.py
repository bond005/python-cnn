# -*- coding: utf-8 -*-
import sys

__all__ = ['convolution_layer', 'subsampling_layer']

if sys.version_info < (3, 4):
    print("CNN requires Python >= 3.4")
    sys.exit(1)
del sys

