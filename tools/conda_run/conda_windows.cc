// Copyright 2019, Intel Corporation.

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/process.hpp>

#include "pmlc/util/env.h"
#include "tools/cpp/runfiles/runfiles.h"

// #define DEBUG

using bazel::tools::cpp::runfiles::Runfiles;
namespace fs = boost::filesystem;
namespace bp = boost::process;

std::string WindowsPath(std::string path) {
  std::replace(path.begin(), path.end(), '/', '\\');
  return path;
}

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

    auto python = fs::canonical(runfiles->Rlocation("com_intel_plaidml_conda_windows/env/python.exe"));
    auto conda_env = WindowsPath(python.parent_path().string());
    auto conda_exe = pmlc::util::getEnvVar("CONDA_EXE");
    if (conda_exe.empty()) {
      conda_exe = R"(C:\tools\Miniconda3\Scripts\conda.exe)";
      if (!fs::exists(conda_exe)) {
        throw std::runtime_error("Missing environment variable: CONDA_EXE");
      }
    }
    auto conda = WindowsPath(fs::canonical(conda_exe).string());

#ifdef DEBUG
    std::cerr << "python: " << python << std::endl;
    std::cerr << "conda_env: " << conda_env << std::endl;
    std::cerr << "conda_exe: " << conda_exe << std::endl;
    std::cerr << "conda: " << conda << std::endl;
#endif

    // Adjust environment variables to activate conda environment
    pmlc::util::setEnvVar("CONDA_DEFAULT_ENV", conda_env);
    pmlc::util::setEnvVar("CONDA_PREFIX", conda_env);

    // Use conda run to adjust the PATH
    bp::opstream os;
    bp::ipstream is;
    std::vector<std::string> args{"run", "-p", conda_env, "path"};
    // std::cerr << "conda run -p $CONDA_ENV path" << std::endl;
    bp::child proc(bp::exe = conda, bp::args = args, bp::std_out > is, bp::std_in < os);
    std::string line;
    while (proc.running() && std::getline(is, line) && !line.empty()) {
      // std::cerr << line << std::endl;
      auto pos = line.find_first_of('=');
      if (pos != std::string::npos) {
        pmlc::util::setEnvVar(line.substr(0, pos), line.substr(pos + 1));
      }
    }
    proc.wait();

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
    args.clear();
    for (int i = 1; i < argc; i++) {
      args.push_back(argv[i]);
    }
    return bp::system(bp::exe = python.string(), bp::args = args);
  } catch (const std::exception& ex) {
    std::cerr << "Caught unhandled exception: " << ex.what() << std::endl;
  } catch (...) {
    std::cerr << "Caught unhandled exception" << std::endl;
  }
  return EXIT_FAILURE;
}
