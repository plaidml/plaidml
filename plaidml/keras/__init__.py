# Copyright Vertex.AI.
#
# Licensed under the GNU Affero General Public License V3 (the License) ;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.gnu.org/licenses/agpl-3.0.en.html
"""Patches in a PlaidML backend for Keras.

This module hooks the system meta module path to add a backend for Keras
that uses PlaidML for computation.  The actual backend is implemented in
backend.py.

To use this module to install the PlaidML backend:

.. code-block:: python

    import plaidml.keras
    plaidml.keras.install_backend()

This should be done in the main program module, after ``__future__`` imports
(if any) and before importing any Keras modules.  Calling ``install()`` replaces
the standard keras.backend module with plaidml.keras.backend, causing subsequently
loaded Keras modules to use PlaidML.

You can explicitly set the installed backend via the environment:
    PLAIDML_KERAS_BACKEND: Selects the backend to use.
                           If this is not set, the standard PlaidML backend is used.
                           Possible values are "plaidml" and "theano".

You can also explicitly pass the backend in the call to ``install_backend()``.

(As an aside: we don't use the standard Keras approach of having you edit
``~/.keras/keras.json`` to set the backend, because we want code that doesn't patch
in the PlaidML backend loader to continue to work.  If Keras ever does support
dynamic loading of backends that aren't hard-coded into Keras, we will switch
to that mechanism.)
"""

# TODO: Update the tracing code to work on older devices.
# For posterity, here's the text:
# You can also enable API tracing by setting an environment variable:
#  PLAIDML_TRACE_FILENAME: Enables tracing, saving the output to the indicated file.

from __future__ import print_function

from six import iteritems

import functools
import importlib
import numpy as np
import os
import sys
import types

_BACKENDS = {'plaidml': '.backend', 'theano': 'keras.backend.theano_backend'}


def install_backend(import_path='keras.backend',
                    backend=os.getenv('PLAIDML_KERAS_BACKEND', 'plaidml'),
                    trace_file=os.getenv('PLAIDML_TRACE_FILENAME')):
    """Installs the PlaidML backend loader, overriding the default keras.backend.

    Args:
        import_path: The name of the module to patch.
        backend: The name of the backend to patch in.
        trace_file: A file object to write trace data to.  This may also be the
                    name of a file, which will be opened with mode 'w' (clobbering
                    the existing file, if any).
    """
    sys.meta_path = [_PlaidMLBackendFinder(import_path, backend, trace_file)] + sys.meta_path

    # Hack around Keras expecting everything not Tensorflow to be Theano.
    from keras.utils import conv_utils
    conv_utils.convert_kernel = lambda x: x


class _PlaidMLBackendFinder(object):

    def __init__(self, repname, backend_name, trace_file):
        self._repname = repname
        self._backend_name = backend_name
        try:
            self._backend_modname = _BACKENDS[backend_name]
        except KeyError:
            raise RuntimeError('Unknown backend \'%s\'; possible values are \'%s\'' %
                               (backend_name, '\', \''.join(_BACKENDS.keys())))
        self._trace_file = trace_file

    def find_module(self, fullname, path=None):
        if fullname != self._repname:
            return None
        tail = fullname.rsplit('.', 1)[-1]
        self._keras_path = [os.path.join(elt, tail) for elt in path]
        return self

    def load_module(self, fullname):
        mod = types.ModuleType(self._repname)
        mod.__path__ = self._keras_path
        sys.modules[fullname] = mod
        self._add_imports(mod, self._backend_modname)
        # self._add_intercepts(mod)
        if self._backend_name != 'plaidml':
            # The included Keras backends require some additional definitions.
            # Note that we don't intercept these.
            self._add_imports(mod, 'keras.backend.common')
            mod.backend = lambda: self._backend_name
        return mod

    def _add_imports(self, mod, import_modname):
        impl = importlib.import_module(import_modname, __name__)
        for (k, v) in iteritems(impl.__dict__):
            setattr(mod, k, v)
        return mod
