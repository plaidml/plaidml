// Copyright 2019 Intel Corporation

#include "base/util/file.h"

#include <boost/format.hpp>

#include "base/util/throw.h"

namespace vertexai {

std::string ReadFile(const boost::filesystem::path& path, bool binary) {
  std::ios_base::openmode mode = std::ios_base::in;
  if (binary) {
    mode |= std::ios::binary;
  }
  std::ifstream ifs(path.string(), mode);
  if (ifs.fail()) {
    throw_with_trace(std::runtime_error(str(boost::format("Unable to open file \"%1%\"") % path)));
  }
  auto it = std::istreambuf_iterator<char>(ifs);
  auto it_end = std::istreambuf_iterator<char>();
  std::string contents(it, it_end);
  if (ifs.bad()) {
    throw_with_trace(std::runtime_error(str(boost::format("Unable to fully read file \"%1%\"") % path)));
  }
  return contents;
}

void WriteFile(const boost::filesystem::path& path,  //
               bool binary,                          //
               const std::function<void(std::ofstream& fout)>& writer) {
  if (path.has_parent_path()) {
    boost::filesystem::create_directory(path.parent_path());
  }
  std::ios_base::openmode mode = std::ios_base::out;
  if (binary) {
    mode |= std::ios::binary;
  }
  std::ofstream fout(path.string(), mode);
  writer(fout);
}

void WriteFile(const boost::filesystem::path& path,  //
               const std::string& contents,          //
               bool binary) {
  WriteFile(path, binary, [contents](std::ofstream& fout) { fout << contents; });
}

}  // namespace vertexai
