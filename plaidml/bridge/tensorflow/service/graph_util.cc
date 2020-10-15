// Copyright 2020 Intel Corporation.

#include "plaidml/bridge/tensorflow/service/graph_util.h"

#include <string>
#include <vector>

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

#include "tensorflow/compiler/tf2xla/graph_compiler_util.h"
#include "tensorflow/core/util/dump_graph.h"

namespace xla {
namespace plaidml {

// Translate Frozen Graph to HLO Module
StatusOr<std::unique_ptr<HloModule>> ImportFrozenGraph(std::string frozen_graph_def_file_path,
                                                       std::vector<std::string> input_names,
                                                       std::vector<std::string> output_names) {
  tensorflow::GraphDef frozen_graph_def;
  if (ReadBinaryProto(tensorflow::Env::Default(), frozen_graph_def_file_path, &frozen_graph_def).ok()) {
    // Temp: dump graph def to file for debug
    // This dumps to TF_DUMP_GRAPH_PREFIX
    DumpGraphDefToFile("cc_frozen_graph_def.pb", frozen_graph_def);
    LocalClient* xla_client = ClientLibrary::LocalClientOrDie();
    XlaComputation xla_computation;
    tensorflow::tf2xla::Config tf2xla_config;
    for (auto input_name : input_names) {
      tf2xla_config.add_feed()->mutable_id()->set_node_name(input_name);
    }
    for (auto output_name : output_names) {
      tf2xla_config.add_fetch()->mutable_id()->set_node_name(output_name);
    }
    if (ConvertGraphDefToXla(frozen_graph_def, tf2xla_config, xla_client, &xla_computation).ok()) {
      const HloModuleProto& module_proto = xla_computation.proto();
      auto module_config =
          HloModule::CreateModuleConfigFromProto(module_proto, xla::GetDebugOptionsFromFlags()).ValueOrDie();
      return HloModule::CreateFromProto(module_proto, module_config).ValueOrDie();
    } else {
      return tensorflow::errors::Internal("Error converting frozen graph to XLA");
    }
  } else {
    return tensorflow::errors::Internal("Error reading frozen graph file: " + frozen_graph_def_file_path);
  }
}

}  // namespace plaidml
}  // namespace xla
