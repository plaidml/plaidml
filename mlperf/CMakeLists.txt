set(MLPERF_LOADGEN_DIR ${mlperf_SOURCE_DIR}/loadgen)

pybind11_add_module(mlperf_loadgen
  ${MLPERF_LOADGEN_DIR}/issue_query_controller.cc
  ${MLPERF_LOADGEN_DIR}/loadgen.cc
  ${MLPERF_LOADGEN_DIR}/logging.cc
  ${MLPERF_LOADGEN_DIR}/test_settings_internal.cc
  ${MLPERF_LOADGEN_DIR}/utils.cc
  ${MLPERF_LOADGEN_DIR}/version.cc
  ${MLPERF_LOADGEN_DIR}/version_generated.cc
  ${MLPERF_LOADGEN_DIR}/bindings/python_api.cc
)

pml_py_library(
  NAME py
  SRCS
    __init__.py
    backend_null.py
    backend_onnxruntime.py
    backend_pytorch_native.py
    backend_pytorch.py
    backend_tf.py
    backend_tflite.py
    backend.py
    coco.py
    dataset.py
    imagenet.py
    main.py
    mlperf.conf
    pycoco.py
    user.conf
    models/anchor_generator.py
    models/base_model_r34.py
    models/convert_tf_weights.py
    models/ssd_mobilenet_v1.py
    models/ssd_r34.py
    models/utils.py
  DEPS
    mlperf_loadgen
)

set(_MLPERF_LOADGEN_MODULE "${PYTHON_MODULE_PREFIX}mlperf_loadgen${PYTHON_MODULE_EXTENSION}")

add_custom_command(
  TARGET mlperf_loadgen
  COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${_MLPERF_LOADGEN_MODULE}
    ${CMAKE_CURRENT_BINARY_DIR}/${_MLPERF_LOADGEN_MODULE}
)

pml_py_wheel(
  NAME wheel
  PKG_NAME "mlperf"
  VERSION 1.1
  ABI cp37m
  PY_VER cp37
  PLATFORM "manylinux1_x86_64" # TODO: support other platforms
  DEPS mlperf_loadgen
  PY_DEPS ::py
)
