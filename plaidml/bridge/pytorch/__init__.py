# Copyright 2019, Intel Corporation.

import contextlib
import os

import plaidml.settings
from plaidml.bridge.pytorch.plaidml_pytorch import *


@contextlib.contextmanager
def toggle():
    try:
        enable(
            device_id=plaidml.settings.get('PLAIDML_DEVICE'),
            target_id=plaidml.settings.get('PLAIDML_TARGET'),
        )
        yield
    finally:
        disable()
