# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

set(_TARGET PlaidMLFuncTests)
set(_LABEL plaidml/bridge/openvino)

addIeTargetTest(
  NAME ${_TARGET}
  ROOT ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDENCIES plaidmlPlugin
  LINK_LIBRARIES IE::funcSharedTests
  ADD_CPPLINT
  LABELS ${_LABEL} test
)

addIeTargetTest(
  NAME ${_TARGET}Smoke
  ROOT ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDENCIES plaidmlPlugin
  DEFINES SMOKE_TESTS_ONLY
  LINK_LIBRARIES IE::funcSharedTests
  ADD_CPPLINT
  LABELS ${_LABEL} smoke
)

add_custom_target(check-bridge-openvino
  COMMAND ${CMAKE_CTEST_COMMAND} -L ${_LABEL}
  USES_TERMINAL
)

add_dependencies(check-bridge-openvino ${_TARGET})

if(PML_BUILD_TESTS)
  pml_add_checks(CHECKS smoke DEPS ${_TARGET}Smoke)
  pml_add_checks(CHECKS test DEPS ${_TARGET})
endif()
