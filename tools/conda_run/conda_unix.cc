// Copyright 2019, Intel Corporation.

#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include "pmlc/util/env.h"
#include "tools/cpp/runfiles/runfiles.h"

// #define DEBUG

using bazel::tools::cpp::runfiles::Runfiles;
namespace fs = boost::filesystem;

extern char** environ;

int main(int argc, char* argv[]) {
  try {
    // Initialize runfiles
    std::string error;
    std::unique_ptr<Runfiles> runfiles(Runfiles::Create(argv[0], &error));
    if (!runfiles) {
      std::cerr << "Failed to load runfiles manifest: " << error << std::endl;
      return EXIT_FAILURE;
    }
    for (const auto& var : runfiles->EnvVars()) {
      pmlc::util::setEnvVar(var.first, var.second);
    }

    // locate assets within runfiles
    auto python = fs::canonical(runfiles->Rlocation("com_intel_plaidml_conda/env/bin/python"));
    auto conda_env = python.parent_path().parent_path().string();

    // Adjust environment variables to activate conda environment
    pmlc::util::setEnvVar("CONDA_DEFAULT_ENV", conda_env);
    auto path = pmlc::util::getEnvVar("PATH");
    auto new_path = python.parent_path().string() + ":" + path;
    pmlc::util::setEnvVar("PATH", new_path);

    // This is to support ASAN on macOS.
    // By default, DYLD_INSERT_LIBRARIES is dropped, probably by SIP (I'm not exactly sure).
    // This is a workaround to avoid the exclusion.
    auto dyld_insert_libs = pmlc::util::getEnvVar("_DYLD_INSERT_LIBRARIES");
    if (dyld_insert_libs.size()) {
      pmlc::util::setEnvVar("DYLD_INSERT_LIBRARIES", dyld_insert_libs);
    }

#ifdef DEBUG
    // Useful debugging code
    std::cerr << "argc: " << argc << std::endl;
    for (int i = 0; i < argc; i++) {
      std::cerr << "arg[" << i << "]: " << argv[i] << std::endl;
    }
    std::map<std::string, std::string> env_map;
    for (char** env = environ; *env; env++) {
      std::string env_str(*env);
      auto pos = env_str.find("=");
      env_map.emplace(env_str.substr(0, pos), env_str.substr(pos + 1));
    }
    for (auto kvp : env_map) {
      std::cerr << kvp.first << "=" << kvp.second << std::endl;
    }
#endif

    // python $argv[1:]
    std::vector<const char*> raw_args{python.string().c_str()};
    for (int i = 1; i < argc; i++) {
      raw_args.push_back(argv[i]);
    }
    raw_args.push_back(nullptr);
    return execv(python.c_str(), const_cast<char* const*>(raw_args.data()));
  } catch (const std::exception& ex) {
    std::cerr << "Caught unhandled exception: " << ex.what() << std::endl;
  } catch (...) {
    std::cerr << "Caught unhandled exception" << std::endl;
  }
  return EXIT_FAILURE;
}
