include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cc
)

file(GLOB HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
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
