// Copyright Vertex.AI

#include "testing/runfiles_db.h"

#include <cstdlib>
#include <fstream>

namespace vertexai {
namespace testing {

RunfilesDB::RunfilesDB(const char* prefix, const char* environ_override_var) {
  // TODO: Once we're on C++17, and have the standard library filesystem APIs
  // available, this code can be simplified.
  if (prefix && *prefix) {
    prefix_ = prefix;
    if (prefix_[prefix_.size()] != '/') {
      prefix_ += '/';
    }
  }

  if (environ_override_var) {
    const char* override = std::getenv(environ_override_var);
    if (override && * override) {
      env_override_ = override;
      if (env_override_[env_override_.size()] != '/') {
        env_override_ += '/';
      }
    }
  }

  if (!env_override_.size()) {
    const char* test_srcdir = std::getenv("TEST_SRCDIR");
    if (test_srcdir) {
      std::string manifest_filename = test_srcdir;
      manifest_filename += "/MANIFEST";
      std::ifstream manifest{manifest_filename};
      while (manifest) {
        std::string logical_name;
        std::string physical_name;
        manifest >> logical_name >> physical_name;
        if (manifest) {
          logical_to_physical_[logical_name] = physical_name;
        }
      }
    }
  }
}

std::string RunfilesDB::operator[](const char* logical_filename) {
  if (env_override_.size()) {
    return env_override_ + logical_filename;
  }
  std::string logical = prefix_ + logical_filename;
  std::lock_guard<std::mutex> lock{mu_};
  auto r = logical_to_physical_.emplace(logical, logical_filename);
  return r.first->second;
}

}  // namespace testing
}  // namespace vertexai
