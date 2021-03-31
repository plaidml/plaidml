// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
  return {
      ".*ShapeOfLayerTest.*",  // Broken until https://github.com/plaidml/openvino/issues/88 is fixed
#ifdef SMOKE_TESTS_ONLY
      "^(?!smoke).*",
#endif  // SMOKE_TESTS_ONLY
  };
}
