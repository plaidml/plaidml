// Copyright 2020 Intel Corporation.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace plaidml {

StatusOr<std::unique_ptr<HloModule>> ImportFrozenGraph(std::string frozen_graph_file_path,
                                                       std::vector<std::string> input_names,
                                                       std::vector<std::string> output_names);

}  // namespace plaidml
}  // namespace xla
