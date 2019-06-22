// Copyright 2019, Intel Corporation.

#ifndef _WIN32
#include <unistd.h>
#endif

#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/process.hpp>

extern char** environ;

namespace fs = boost::filesystem;

#ifdef _WIN32
const char kPathSeparator = ';';
#else
const char kPathSeparator = ':';
#endif

std::string read_file(const fs::path& path) {
  std::ifstream ifs(path.string());
  auto it = std::istreambuf_iterator<char>(ifs);
  std::string contents(it, std::istreambuf_iterator<char>());
  return contents;
}

int main(int argc, char* argv[]) {
  fs::path this_path = fs::absolute(argv[0]);
  if (fs::is_symlink(this_path)) {
    this_path = fs::read_symlink(this_path);
  }
  auto main_str = read_file(fs::canonical(this_path).replace_extension(".main"));
  auto cwd_path = fs::current_path();
  auto main_path = cwd_path / main_str;
  auto runfiles_dir = this_path.replace_extension(".runfiles");
  auto cenv_dir = runfiles_dir / ".cenv";
#ifdef _WIN32
  auto python = cenv_dir / "python.exe";
#else
  auto bin_dir = cenv_dir / "bin";
  auto python = bin_dir / "python";
#endif

  std::vector<std::string> args = {python.string(), main_path.string()};
  for (int i = 1; i < argc; i++) {
    args.push_back(argv[i]);
  }

  std::map<std::string, std::string> env_map;
  for (char** env = environ; *env; env++) {
    std::string env_str(*env);
    auto pos = env_str.find("=");
    env_map.emplace(env_str.substr(0, pos), env_str.substr(pos + 1));
  }

  std::list<std::string> python_paths;
  for (const auto& dir : fs::directory_iterator(runfiles_dir)) {
    if (fs::is_directory(dir) && dir != cenv_dir) {
      python_paths.emplace_back(dir.path().string());
      // This whole mess is due to: https://github.com/bazelbuild/bazel/issues/4594
      auto external_dir = dir.path() / "external";
      if (fs::is_directory(external_dir)) {
        for (const auto& inner_dir : fs::directory_iterator(external_dir)) {
          if (fs::is_directory(inner_dir)) {
            python_paths.emplace_front(inner_dir.path().string());
          }
        }
      }
    }
  }

  std::stringstream ss;
  bool is_first = true;
  for (const auto& dir : python_paths) {
    if (is_first) {
      is_first = false;
    } else {
      ss << kPathSeparator;
    }
    ss << dir;
  }

  std::string python_path = ss.str();
  auto it = env_map.find("PYTHONPATH");
  if (it != env_map.end()) {
    python_path = ss.str() + kPathSeparator + it->second;
  }

  env_map["PYTHONPATH"] = python_path;
  env_map["RUNFILES_DIR"] = runfiles_dir.string();
  env_map["CONDA_DEFAULT_ENV"] = cenv_dir.string();
#ifdef _WIN32
  auto scripts_dir = cenv_dir / "Scripts";
  auto bin_dir = cenv_dir / "Library" / "bin";
  env_map["PATH"] = bin_dir.string() + kPathSeparator +      //
                    cenv_dir.string() + kPathSeparator +     //
                    scripts_dir.string() + kPathSeparator +  //
                    env_map["PATH"];
#else
  env_map["PATH"] = bin_dir.string() + kPathSeparator +  //
                    env_map["PATH"];
#endif

  if (env_map.count("ASAN_ENABLE")) {
    env_map["DYLD_INSERT_LIBRARIES"] =
        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/10.0.1/lib/"
        "darwin/libclang_rt.asan_osx_dynamic.dylib";
  }

  // for (const auto& kvp : env_map) {
  //   std::cout << kvp.first << "=" << kvp.second << std::endl;
  // }

#ifdef _WIN32
  namespace bp = boost::process;
  bp::environment new_env;
  for (const auto& kvp : env_map) {
    new_env[kvp.first] = kvp.second;
  }
  args.erase(args.begin());
  return bp::system(bp::exe = python.string(), bp::args = args, new_env);
#else
  std::vector<const char*> raw_args;
  for (const auto& arg : args) {
    raw_args.push_back(arg.c_str());
  }
  raw_args.push_back(nullptr);

  std::vector<std::string> env_strs;
  std::vector<const char*> raw_envs;
  for (const auto& kvp : env_map) {
    auto it = env_strs.insert(env_strs.end(), kvp.first + "=" + kvp.second);
    raw_envs.push_back(it->c_str());
  }
  raw_envs.push_back(nullptr);

  return execve(python.c_str(),                             //
                const_cast<char* const*>(raw_args.data()),  //
                const_cast<char* const*>(raw_envs.data()));
#endif
}
