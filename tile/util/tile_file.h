#pragma once

#include <string>
#include <vector>

#include "base/util/zipfile.h"
#include "tile/proto/metadata.pb.h"

namespace vertexai {
namespace tile {

class TileFile {
 public:
  explicit TileFile(const std::string& path);

  metadata::proto::Metadata ReadMetadata();
  std::vector<float> GetTensorFloatData(const metadata::proto::Tensor& tensor);

 private:
  UnZipArchive archive_;
};

}  // namespace tile
}  // namespace vertexai
