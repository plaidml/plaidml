# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

addIeTargetTest(
  NAME PlaidMLFuncTests
  ROOT ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDENCIES plaidmlPlugin
  LINK_LIBRARIES IE::funcSharedTests
  ADD_CPPLINT
  LABELS OpenVINO
)

add_dependencies(check-test PlaidMLFuncTests)
