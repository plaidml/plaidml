import numpy as np

from plaidml.bridge.tensorflow.tests import archive_py_generated as schema

dataTypeMap = {
    'int8': (schema.I8DataT, schema.Data.I8Data),
    'int16': (schema.I16DataT, schema.Data.I16Data),
    'int32': (schema.I32DataT, schema.Data.I32Data),
    'int64': (schema.I64DataT, schema.Data.I64Data),
    'uint8': (schema.U8DataT, schema.Data.U8Data),
    'uint16': (schema.U16DataT, schema.Data.U16Data),
    'uint32': (schema.U32DataT, schema.Data.U32Data),
    'uint64': (schema.U64DataT, schema.Data.U64Data),
    'float16': (schema.F16DataT, schema.Data.F16Data),
    'float32': (schema.F32DataT, schema.Data.F32Data),
    'float64': (schema.F64DataT, schema.Data.F64Data),
}


def convertBuffer(name, src):
    dtype = dataTypeMap.get(str(src.dtype))
    if not dtype:
        raise Exception('Unknown dtype: {}'.format(src.dtype))
    data = dtype[0]()
    data.data = np.ndarray.flatten(src)
    buffer = schema.BufferT()
    buffer.name = name
    buffer.dataType = dtype[1]
    buffer.data = data
    return buffer
