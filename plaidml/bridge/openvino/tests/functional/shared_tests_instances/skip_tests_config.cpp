// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
  return {
      ".*ShapeOfLayerTest.*",  // Broken until https://github.com/plaidml/openvino/issues/88 is fixed
      "smoke3In.*",
      "smoke5In.*",
      "DetectionOutput3In.*",
      "EmbeddingSegmentsSum.*",
      "smoke/EmbeddingSegmentsSumLayerTest.*",
      "smoke/NmsLayerTest.CompareWithRefs.*",
      "smoke/ProposalLayerTest.*",
      "smoke_PSROIPooling.*",
      "smoke/RegionYoloLayerTest.*",
      "smoke_TestsROIAlign_max.*",
      "smoke/ROIPoolingLayerTest.*",
#ifdef SMOKE_TESTS_ONLY
      "^(?!smoke).*",
#endif  // SMOKE_TESTS_ONLY
  };
}
