// Copyright 2018 Intel Corporation

#include "base/util/zipfile.h"

#include <algorithm>
#include <stdexcept>

const size_t BLOCK_SIZE = 8 * 1024;

namespace vertexai {

UnZipArchive::UnZipArchive(const std::string& path) : zip_file_(unzOpen64(path.c_str())) {
  if (!zip_file_) {
    throw std::runtime_error("Cannot open zip archive for extraction.");
  }
}

UnZipArchive::~UnZipArchive() { unzClose(zip_file_); }

UnZipFile UnZipArchive::OpenFile(const std::string& filename) { return UnZipFile(zip_file_, filename); }

UnZipFile::UnZipFile(unzFile zip_file, const std::string& filename) : zip_file_(zip_file) {
  if (unzLocateFile(zip_file_, filename.c_str(), nullptr) != UNZ_OK) {
    auto msg = std::string("Could not locate file within zip archive: ") + filename;
    throw std::runtime_error(msg);
  }
  unzOpenCurrentFile(zip_file_);
  unzGetCurrentFileInfo64(zip_file_, &fi_, nullptr, 0, nullptr, 0, nullptr, 0);
}

UnZipFile::~UnZipFile() { unzCloseCurrentFile(zip_file_); }

std::string UnZipFile::ReadString() {
  std::string str(fi_.uncompressed_size, '\0');
  ReadInto(&str[0], str.size());
  return str;
}

void UnZipFile::ReadInto(void* buf, std::size_t len) {
  char* ptr = static_cast<char*>(buf);
  std::size_t bytes_remaining = len;
  while (bytes_remaining) {
    std::size_t bytes_to_read = std::min(BLOCK_SIZE, bytes_remaining);
    int err = unzReadCurrentFile(zip_file_, ptr, bytes_to_read);
    ptr += bytes_to_read;
    bytes_remaining -= bytes_to_read;
    if (err < 0) {
      throw std::runtime_error("Failed to read file within zip archive.");
    }
  }
}

}  // namespace vertexai
