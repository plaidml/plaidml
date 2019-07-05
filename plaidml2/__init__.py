# Copyright 2019 Intel Corporation.

import sys

from plaidml2.core import *
from plaidml2.core import settings

# allow core attributes to be seen from top level module
__version__ = core.__version__

# allow core modules to be imported from the top level package
sys.modules[__name__ + '.settings'] = settings
