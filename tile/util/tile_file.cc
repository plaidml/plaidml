#include "tile/util/tile_file.h"

#include <google/protobuf/util/json_util.h>

#include <algorithm>
#include <memory>
#include <regex>
#include <vector>

#include "tile/lang/parser.h"
#include "tile/proto/tile.pb.h"

namespace vertexai {
namespace tile {
namespace util {

namespace {

std::shared_ptr<lang::TensorValue> ReadTensor(UnZipArchive* zip_file, const std::string& name) {
  auto tensor_file = zip_file->OpenFile(name);

  uint64_t shape_size;
  tensor_file.ReadInto(&shape_size, sizeof(shape_size));

  std::string shape_buf(shape_size, '\0');
  tensor_file.ReadInto(&shape_buf[0], shape_buf.size());

  proto::TensorShape pb_shape;
  pb_shape.ParseFromString(shape_buf);

  auto tensor_shape = FromProto(pb_shape);

  auto buffer = std::make_shared<SimpleBuffer>();
  buffer->bytes.resize(tensor_shape.byte_size());
  tensor_file.ReadInto(buffer->bytes.data(), buffer->bytes.size());

  return lang::TensorValue::make(buffer, tensor_shape, true);
}

// This is a dummy buffer to satisfy the BoundFunction and FunctionApplication.
// In this case, we have no need for allocating actual buffers.
struct NullBuffer : lang::BufferBase {};

std::shared_ptr<lang::TensorValue> MakeTensor(TensorShape shape) {
  auto null_buffer = std::make_shared<NullBuffer>();
  return lang::TensorValue::make(null_buffer, shape, false);
}

}  // namespace

TileFile::TileFile(const std::string& path) : archive_(path) {}

lang::RunInfo TileFile::Load() {
  auto metadata = ReadMetadata();
  auto code = archive_.OpenFile("code").ReadString();

  // This code was lifted from plaidml.cc/plaidml_load_function().
  lang::Parser parser;
  auto dexified = DeXify(parser.Parse(code));

  // Unfortunately, we don't serialize the number of temps (which is needed to do inlining)
  // So we recompute that here, based on the fact that all temps start with _T (otherwise reserved)
  for (const lang::Op& op : dexified.ops) {
    if (op.output.size() >= 2 && op.output[0] == '_' && op.output[1] == 'T') {
      int count = std::atoi(op.output.substr(2, op.output.size() - 2).c_str()) + 1;
      dexified.next_tmp = std::max(dexified.next_tmp, static_cast<uint64_t>(count));
    }
  }

  std::vector<std::shared_ptr<lang::TensorValue>> bound_inputs;
  for (const auto& input : dexified.inputs) {
    if (input.name[0] == '_') {
      bound_inputs.push_back(ReadTensor(&archive_, "data_" + input.name));
    }
  }

  auto bound = std::make_shared<lang::BoundFunction>(dexified, bound_inputs);

  // This code was derived from the following functions in plaidml.cc:
  //  plaidml_alloc_invoker
  //  plaidml_set_invoker_input
  //  plaidml_set_invoker_output
  //  plaidml_schedule_invocation
  // The reason this code was not directly used from plaidml.cc is that in this case,
  // we have no need for binding to actual buffers or dealing with devices.

  lang::FunctionApplication applier(bound);

  auto num_inputs = bound->num_inputs();
  for (size_t i = 0; i < num_inputs; i++) {
    std::string input_name = bound->input_name(i);
    auto shape = FromProto(metadata.inputs().at(input_name));
    for (auto& dim : shape.dims) {
      if (dim.size == 0) {
        dim.size = 1;
      }
    }
    applier.SetInput(input_name, MakeTensor(shape));
  }

  applier.SetDone();

  auto num_outputs = bound->num_outputs();

  lang::BoundFunction composer;
  composer.AddDependency(applier);
  for (size_t i = 0; i < num_outputs; i++) {
    std::string name = bound->output_name(i);
    auto shape = applier.GetOutputShape(name);
    composer.AddUpdate(MakeTensor(shape), applier.GetOutput(name));
  }
  composer.Done();
  return composer.PrepareToRun();
}

metadata::proto::Metadata TileFile::ReadMetadata() {
  auto metadata_file = archive_.OpenFile("metadata");
  // This is a backwards compatiblity hack
  std::regex dims_re("dimensions");
  auto metadata_coded = std::regex_replace(metadata_file.ReadString(), dims_re, "dims");
  metadata::proto::Metadata metadata;
  if (!::google::protobuf::util::JsonStringToMessage(metadata_coded, &metadata).ok()) {
    throw std::runtime_error("Unable to parse benchmark metadata");
  }

  return metadata;
}

std::vector<float> TileFile::GetTensorFloatData(const metadata::proto::Tensor& tensor) {
  std::vector<float> result;
  if (tensor.filename().empty()) {
    throw std::runtime_error{"No internal data or data filename found in tensor"};
  }
  const auto& d0 = tensor.shape().dims()[0];
  std::size_t size = d0.stride() * d0.size();
  result.resize(size);

  auto tensor_file = archive_.OpenFile(tensor.filename());
  tensor_file.ReadInto(result.data(), size * sizeof(float));

  return result;
}

}  // namespace util
}  // namespace tile
}  // namespace vertexai
