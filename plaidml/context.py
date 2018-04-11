# Copyright Vertex.AI

import ctypes
import json


class Context(object):

    def __init__(self, lib):
        self._as_parameter_ = lib.vai_alloc_ctx()
        if not self._as_parameter_:
            raise MemoryError('PlaidML operation context')
        self._free = lib.vai_free_ctx
        self._cancel = lib.vai_cancel_ctx
        self._set_eventlog = lib.vai_set_eventlog

    def __del__(self):
        self.shutdown()

    def cancel(self):
        self._cancel(self)

    def set_eventlog_filename(self, filename):
        config = {
            '@type': 'type.vertex.ai/vertexai.eventing.file.proto.EventLog',
            'filename': filename
        }
        self._set_eventlog(self, json.dumps(config).encode())

    def shutdown(self):
        if hasattr(self, '_free') and self._as_parameter_:
            self._free(self)
        self._as_parameter_ = None
