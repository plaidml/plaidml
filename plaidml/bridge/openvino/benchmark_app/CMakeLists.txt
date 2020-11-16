set(_SRC_DIR ${openvino_SOURCE_DIR}/inference-engine/samples/benchmark_app)
set(_SRCS
  ${_SRC_DIR}/inputs_filling.cpp
  ${_SRC_DIR}/main.cpp
  ${_SRC_DIR}/statistics_report.cpp
  ${_SRC_DIR}/utils.cpp
)

pml_cc_binary(
  NAME benchmark_app
  SRCS ${_SRCS}
  DEPS
    IE::inference_engine
    format_reader
    gflags
)
