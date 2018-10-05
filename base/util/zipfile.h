// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <unzip.h>

#include <string>

namespace vertexai {

class UnZipFile {
  friend class UnZipArchive;

 private:
  UnZipFile(unzFile zip_file, const std::string& filename);

 public:
  ~UnZipFile();

  std::string ReadString();
  void ReadInto(void* buf, std::size_t len);

 private:
  unzFile zip_file_;
  unz_file_info64 fi_;
};

class UnZipArchive {
 public:
  explicit UnZipArchive(const std::string& path);
  ~UnZipArchive();

  UnZipFile OpenFile(const std::string& filename);

 private:
  unzFile zip_file_;
};

}  // namespace vertexai
