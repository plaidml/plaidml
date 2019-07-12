# Copyright 2019 Intel Corporation.

import logging
import os
import sys

logger = logging.getLogger(__name__)

if os.getenv('PLAIDML_VERBOSE'):
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.debug('PLAIDML_VERBOSE enabled on logger: {}'.format(logger.name))

from plaidml2.core import *
from plaidml2.core import settings

# allow core attributes to be seen from top level module
__version__ = core.__version__

# allow core modules to be imported from the top level package
sys.modules[__name__ + '.settings'] = settings
