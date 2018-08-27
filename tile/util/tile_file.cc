#include "tile/util/tile_file.h"

#include <google/protobuf/util/json_util.h>

namespace vertexai {
namespace tile {

TileFile::TileFile(const std::string& path) : archive_(path) {}

tile::metadata::proto::Metadata TileFile::ReadMetadata() {
  auto metadata_file = archive_.OpenFile("metadata");
  auto metadata_coded = metadata_file.ReadString();
  tile::metadata::proto::Metadata metadata;
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
  const auto& d0 = tensor.shape().dimensions()[0];
  std::size_t size = d0.stride() * d0.size();
  result.resize(size);

  auto tensor_file = archive_.OpenFile(tensor.filename());
  tensor_file.ReadInto(result.data(), size * sizeof(float));

  return result;
}

}  // namespace tile
}  // namespace vertexai
