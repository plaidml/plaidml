// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
  return {
      ".*ShapeOfLayerTest.*",         // Broken until https://github.com/plaidml/openvino/issues/88 is fixed
      "smoke_PSROIPooling.*",         // Fails tile_to_pxa conversion (corrupt gather interpolation)
      "PSROIPooling.*",               // Fails tile_to_pxa conversion (corrupt gather interpolation)
      "smoke_TestsROIAlign_max.*",    // Fails tile_to_pxa conversion (corrupt gather interpolation)
      "smoke/ROIPoolingLayerTest.*",  // Fails tile_to_pxa conversion (corrupt gather interpolation)
      // The following fail due to the same bad scatter lowering
      "smoke3In.*",
      "smoke5In.*",
      "DetectionOutput3In.*",
      "EmbeddingSegmentsSum.*",
      "smoke/EmbeddingSegmentsSumLayerTest.*",
      "smoke/NmsLayerTest.CompareWithRefs.*",
      "NMS/NmsLayerTest.*",
      "noClipNMS/ProposalLayerTest.*",
      "clipNMS/ProposalLayerTest.*",
      "smoke/ProposalLayerTest.*",
      "tfDecode/ProposalLayerTest.*",
      "smoke/RegionYoloLayerTest.*",
      "TestsRegionYolo.*",
#ifdef SMOKE_TESTS_ONLY
      "^(?!smoke).*",
#endif  // SMOKE_TESTS_ONLY
  };
}
