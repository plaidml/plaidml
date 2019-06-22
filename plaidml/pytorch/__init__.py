# Copyright 2019, Intel Corporation.

import contextlib

from plaidml.pytorch.plaidml_pytorch import *


@contextlib.contextmanager
def toggle():
    try:
        enable()
        yield
    finally:
        disable()
