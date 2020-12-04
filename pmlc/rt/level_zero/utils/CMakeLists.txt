file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/harness/*.cpp
)

file(GLOB HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils/.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include/test_harness/.hpp
)

pml_cc_library(
    NAME lz_wrapper
    HDRS
      ${HEADERS}
    SRCS
      ${SOURCES}
    DEPS
      LevelZero::LevelZero
)
