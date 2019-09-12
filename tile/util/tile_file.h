#pragma once

#include <memory>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include "base/util/zipfile.h"
#include "tile/lang/runinfo.h"
#include "tile/proto/metadata.pb.h"

namespace vertexai {
namespace tile {
namespace util {

struct SimpleBuffer : lang::BufferBase {
  std::vector<uint8_t> bytes;
};

class TileFile {
 public:
  explicit TileFile(const boost::filesystem::path& path);

  lang::RunInfo Load(const std::vector<std::shared_ptr<SimpleBuffer>>& inputs = {});
  metadata::proto::Metadata ReadMetadata();
  std::vector<float> GetTensorFloatData(const metadata::proto::Tensor& tensor);

 private:
  UnZipArchive archive_;
  boost::filesystem::path path_;
};

}  // namespace util
}  // namespace tile
}  // namespace vertexai
