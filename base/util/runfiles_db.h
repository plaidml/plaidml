// Copyright 2018 Intel Corporation

#pragma once

#include <mutex>
#include <string>
#include <unordered_map>

namespace vertexai {

// RunfilesDB maps logical runfiles filenames to physical filenames for tests.
//
// Tests may have data dependencies, defined by the 'data' parameter passed to
// the rules that define them and their library dependencies.  On systems that
// support symlinks, the test environment builds a symlink farm mapping the
// logical data filenames to the physical source files.  On systems that don't
// support symlinks, the test code is responsible for implementing this mapping,
// based on the MANIFEST file created by the test environment.
//
// This class handles MANIFEST parsing and the mapping to physical filenames.
// It works regardless of whether the test environment provides symlinks or not.
class RunfilesDB {
 public:
  // Construct the RunfilesDB.
  //
  // The supplied prefix will be prepended to all lookup requests.
  // For example: db["name"] would become MANIFEST[prefix + "/name"]
  //
  // If the environment variable override is non-nullptr, and the corresponding
  // environment variable is set, the variable's value will be used as the directory
  // for the physical files, regardless of the prefix.
  // For example: db["name"] would become env[var] + "/name"
  explicit RunfilesDB(const char* prefix = nullptr, const char* environ_override_var = nullptr);

  // Maps a logical filename to the corresponding physical filename.
  std::string operator[](const char* logical_filename);

 private:
  std::mutex mu_;
  std::string prefix_;
  std::string relative_prefix_;
  std::string env_override_;
  std::unordered_map<std::string, std::string> logical_to_physical_;
};

}  // namespace vertexai
