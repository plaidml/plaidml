// Copyright 2018 Intel Corporation.

#include "testing/plaidml_config.h"
#include "base/util/env.h"

namespace vertexai {
namespace testing {

const char* PlaidMLConfig() {
  return R"(
{
    "platform": {
        "@type": "type.vertex.ai/vertexai.tile.local_machine.proto.Platform",
        "hardware_configs": []
    }
}
  )";
}

}  // namespace testing
}  // namespace vertexai
