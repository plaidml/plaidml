// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
  return {
      ".*Acos.*FP16.*",               // TODO
      ".*Asin.*FP16.*",               // TODO
      ".*Atan.*FP16.*",               // TODO
      ".*Ceiling.*FP16.*",            // TODO
      ".*Cosh.*FP16.*",               // TODO
      ".*Erf.*FP16.*",                // TODO
      ".*Floor.*FP16.*",              // TODO
      ".*Lrn.*FP16.*",                // TODO
      ".*Sinh.*FP16.*",               // TODO
      ".*Tan.*FP16.*",                // TODO
      ".*Convert.*targetPRC=FP16.*",  // TODO
#ifdef SMOKE_TESTS_ONLY
      "^(?!smoke).*",
#endif  // SMOKE_TESTS_ONLY
  };
}
