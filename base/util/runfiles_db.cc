// Copyright 2018 Intel Corporation

#include "base/util/runfiles_db.h"

#include <cstdlib>
#include <fstream>

#include "base/util/env.h"

namespace vertexai {

RunfilesDB::RunfilesDB(const char* prefix, const char* environ_override_var) {
  // TODO: Once we're on C++17, and have the standard library filesystem APIs
  // available, this code can be simplified.
  if (prefix && *prefix) {
    prefix_ = prefix;
    if (prefix_[prefix_.size()] != '/') {
      prefix_ += '/';
    }
    relative_prefix_ = prefix_.substr(prefix_.find('/') + 1);
  }

  if (environ_override_var) {
    auto override_value = env::Get(environ_override_var);
    if (override_value.length()) {
      env_override_ = override_value;
      if (env_override_[env_override_.size()] != '/') {
        env_override_ += '/';
      }
    }
  }

  if (!env_override_.size()) {
    auto runfiles_dir = env::Get("RUNFILES_DIR");
    if (runfiles_dir.length()) {
      std::string manifest_filename = runfiles_dir;
      manifest_filename += "/MANIFEST";
      std::ifstream manifest{manifest_filename};
      for (;;) {
        std::string line;
        std::getline(manifest, line);
        if (!manifest) {
          break;
        }
        auto split_pos = line.find(' ');
        if (split_pos == std::string::npos) {
          continue;
        }
        std::string logical_name = line.substr(0, split_pos);
        std::string physical_name = line.substr(split_pos + 1);
        logical_to_physical_[logical_name] = physical_name;
      }
    }
  }
}

std::string RunfilesDB::operator[](const char* logical_filename) {
  if (env_override_.size()) {
    return env_override_ + logical_filename;
  }
  std::string logical = prefix_ + logical_filename;
  std::string relative_filename = relative_prefix_ + logical_filename;
  std::lock_guard<std::mutex> lock{mu_};
  auto r = logical_to_physical_.emplace(logical, relative_filename);
  return r.first->second;
}

}  // namespace vertexai
