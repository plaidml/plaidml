# Copyright 2019, Intel Corporation.

import contextlib
import os

import plaidml2.settings
from plaidml2.bridge.pytorch.plaidml_pytorch import *


@contextlib.contextmanager
def toggle():
    try:
        enable(
            device_id=plaidml2.settings.get('PLAIDML_DEVICE'),
            target_id=plaidml2.settings.get('PLAIDML_TARGET'),
        )
        yield
    finally:
        disable()
