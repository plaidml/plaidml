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
      "smoke_Set1.*",
      "smoke_Set2/GatherNDLayerTest.*",
      "smoke/NmsLayerTest.CompareWithRefs.*",
      "smoke/PriorBoxLayerTest.*",
      "smoke/ProposalLayerTest.*",
      "smoke_PSROIPooling.*",
      "smoke/RegionYoloLayerTest.*",
      "smoke_TestsROIAlign_max.*",
      "smoke/ROIPoolingLayerTest.*",
      "smoke_ShuffleChannels4D.*",
      "smoke_ShuffleChannelsNegativeAxis4D.*",
      "smoke/SqueezeUnsqueezeLayerTest.*",
#ifdef SMOKE_TESTS_ONLY
      "^(?!smoke).*",
#endif  // SMOKE_TESTS_ONLY
  };
}
