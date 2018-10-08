#pragma once

#include <string>
#include <vector>

#include "base/util/zipfile.h"
#include "tile/lang/compose.h"
#include "tile/proto/metadata.pb.h"

namespace vertexai {
namespace tile {
namespace util {

class TileFile {
 public:
  explicit TileFile(const std::string& path);

  lang::RunInfo Load();
  metadata::proto::Metadata ReadMetadata();
  std::vector<float> GetTensorFloatData(const metadata::proto::Tensor& tensor);

 private:
  UnZipArchive archive_;
};

}  // namespace util
}  // namespace tile
}  // namespace vertexai
