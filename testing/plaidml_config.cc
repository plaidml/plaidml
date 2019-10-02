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
        "hardware_configs": [
            {
                "description": "CPU (via LLVM) settings",
                "sel": {
                    "and": {
                        "sel": [
                            {
                                "name_regex": "LLVM CPU"
                            },
                            {
                                "vendor_regex": "LLVM"
                            }
                        ]
                    }
                },
                "settings": {
                    "vec_size": 32,
                    "mem_width": 4096,
                    "use_global": true,
                    "max_regs": 8192,
                    "max_mem": 8192,
                    "goal_flops_per_byte": 20,
                    "stripe_config": "llvm_cpu",
                    "use_stripe": true
                }
            }
        ]
    }
}
  )";
}

}  // namespace testing
}  // namespace vertexai
