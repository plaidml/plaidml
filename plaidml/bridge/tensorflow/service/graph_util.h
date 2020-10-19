// Copyright 2020 Intel Corporation.

#pragma once

#include <memory>

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/graph/graph.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

using llvm::ArrayRef;
using llvm::StringRef;

namespace xla {
namespace plaidml {

StatusOr<tensorflow::GraphDef> FreezeGraph(StringRef path, ArrayRef<StringRef> input_names,
                                           ArrayRef<StringRef> output_names);

StatusOr<std::unique_ptr<HloModule>> ImportFrozenGraph(StringRef frozen_graph_file_path,
                                                       ArrayRef<StringRef> input_names,
                                                       ArrayRef<StringRef> output_names);

StatusOr<std::unique_ptr<HloModule>> LowerFrozenGraphToHlo(tensorflow::GraphDef* frozen_graph_def,
                                                           ArrayRef<StringRef> input_names,
                                                           ArrayRef<StringRef> output_names);

}  // namespace plaidml
}  // namespace xla
