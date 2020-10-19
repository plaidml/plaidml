// Copyright 2020 Intel Corporation.

#include "plaidml/bridge/tensorflow/service/graph_util.h"

#include <string>
#include <unordered_set>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/tools/freeze_saved_model.h"
#include "tensorflow/compiler/tf2xla/graph_compiler_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/dump_graph.h"

namespace xla {
namespace plaidml {

// Freeze graph from file path
StatusOr<tensorflow::GraphDef> FreezeGraph(StringRef path, ArrayRef<StringRef> input_names,
                                           ArrayRef<StringRef> output_names) {
  // Load saved model
  tensorflow::SessionOptions session_options;
  auto run_options = tensorflow::RunOptions();
  std::unordered_set<std::string> tags({"train"});
  auto bundle = tensorflow::SavedModelBundle();
  if (tensorflow::LoadSavedModel(session_options, run_options, path.str(), tags, &bundle).ok()) {
    std::unordered_set<std::string> inputs(input_names.size());
    std::unordered_set<std::string> outputs(output_names.size());
    for (auto input_name : input_names) {
      inputs.insert(input_name.str() + ":0");
    }
    for (auto output_name : output_names) {
      outputs.insert(output_name.str() + ":0");
    }
    // Freeze graph from this saved model
    tensorflow::GraphDef frozen_graph_def;
    if (tensorflow::FreezeSavedModel(bundle, &frozen_graph_def, &inputs, &outputs).ok()) {
      return frozen_graph_def;
    } else {
      return tensorflow::errors::Internal("Error freezing graph def");
    }
  } else {
    return tensorflow::errors::Internal("Unable to load saved model");
  }
}

// Translate Frozen Graph to HLO Module
StatusOr<std::unique_ptr<HloModule>> ImportFrozenGraph(StringRef frozen_graph_def_file_path,
                                                       ArrayRef<StringRef> input_names,
                                                       ArrayRef<StringRef> output_names) {
  tensorflow::GraphDef frozen_graph_def;
  if (ReadBinaryProto(tensorflow::Env::Default(), frozen_graph_def_file_path.str(), &frozen_graph_def).ok()) {
    return LowerFrozenGraphToHlo(&frozen_graph_def, input_names, output_names).ValueOrDie();
  } else {
    return tensorflow::errors::Internal("Error reading frozen graph file: " + frozen_graph_def_file_path.str());
  }
}

// Translate frozen graph to HLO Module
StatusOr<std::unique_ptr<HloModule>> LowerFrozenGraphToHlo(tensorflow::GraphDef* frozen_graph_def,
                                                           ArrayRef<StringRef> input_names,
                                                           ArrayRef<StringRef> output_names) {
  // This dumps a human-readable frozen graph to TF_DUMP_GRAPH_PREFIX if specified
  DumpGraphDefToFile("cc_graph_util_frozen_graph_def", *frozen_graph_def);
  LocalClient* xla_client = ClientLibrary::LocalClientOrDie();
  XlaComputation xla_computation;
  tensorflow::tf2xla::Config tf2xla_config;
  for (auto input_name : input_names) {
    tf2xla_config.add_feed()->mutable_id()->set_node_name(input_name.str());
  }
  for (auto output_name : output_names) {
    tf2xla_config.add_fetch()->mutable_id()->set_node_name(output_name.str());
  }
  if (ConvertGraphDefToXla(*(frozen_graph_def), tf2xla_config, xla_client, &xla_computation).ok()) {
    const HloModuleProto& module_proto = xla_computation.proto();
    auto module_config =
        HloModule::CreateModuleConfigFromProto(module_proto, xla::GetDebugOptionsFromFlags()).ValueOrDie();
    return HloModule::CreateFromProto(module_proto, module_config).ValueOrDie();
  } else {
    return tensorflow::errors::Internal("Error converting frozen graph to XLA");
  }
}

}  // namespace plaidml
}  // namespace xla
