// Copyright 2019, Intel Corporation

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include "base/config/config.h"
#include "base/util/file.h"
#include "base/util/logging.h"
#include "base/util/throw.h"
#include "tile/codegen/codegen.pb.h"

namespace fs = boost::filesystem;
namespace po = boost::program_options;

namespace vertexai {
namespace tools {

static const char* kTemplate{R"(#pragma once

#include <string>

%1%
std::string %2%{"%3%", %4%};
%5%
)"};

void gencfg(const po::variables_map& args) {
  tile::codegen::proto::Configs master;

  const auto& srcs_args = args["srcs"];
  if (!srcs_args.empty()) {
    const auto& srcs = srcs_args.as<std::vector<fs::path>>();
    for (const auto& src : srcs) {
      auto json = ReadFile(src);
      IVLOG(1, src << ": " << json);
      auto configs = ParseConfig<tile::codegen::proto::Configs>(json);
      for (const auto& config : configs.configs()) {
        if (master.configs().count(config.first)) {
          throw std::runtime_error(str(boost::format("Duplicate config entry detected: %1%") % config.first));
        }
        master.mutable_configs()->insert(config);
      }
    }
  }

  std::string master_buf;
  master.SerializeToString(&master_buf);

  std::ostringstream body;
  for (const auto ch : master_buf) {
    auto byte = static_cast<unsigned>(static_cast<unsigned char>(ch));
    body << "\\x" << std::hex << std::setw(2) << std::setfill('0') << byte << std::dec;
  }

  std::vector<std::string> identifier_parts;
  auto identifier_full = args["identifier"].as<std::string>();
  boost::split(identifier_parts, identifier_full, boost::is_any_of("::"), boost::token_compress_on);
  IVLOG(1, "parts: " << identifier_parts);

  auto identifier = identifier_parts.back();
  std::ostringstream ns_begin;
  std::ostringstream ns_end;
  for (size_t i = 0; i < identifier_parts.size() - 1; i++) {
    ns_begin << "namespace " << identifier_parts[i] << " {\n";
    ns_end << "\n} // namespace " << identifier_parts[i];
  }

  std::string output{
      str(boost::format(kTemplate) % ns_begin.str() % identifier % body.str() % master_buf.size() % ns_end.str())};
  WriteFile(args["out"].as<fs::path>(), output, true);
}

}  // namespace tools
}  // namespace vertexai

int main(int argc, char* argv[]) {
  try {
    START_EASYLOGGINGPP(argc, argv);

    po::positional_options_description pos_opts;
    pos_opts.add("srcs", -1);

    po::options_description opts{"Allowed options"};
    opts.add_options()                                                           //
        ("help,h", "produce help message")                                       //
        ("verbose,v", po::value<int>()->default_value(0), "increase verbosity")  //
        ("srcs", po::value<std::vector<fs::path>>(), ".json input paths")        //
        ("identifier",                                                           //
         po::value<std::string>()->required(),                                   //
         "specify the identifier (can include namespace)")                       //
        ("out", po::value<fs::path>()->required(), "output path");

    auto parser = po::command_line_parser(argc, argv).options(opts).positional(pos_opts);

    po::variables_map args;
    po::store(parser.run(), args);
    if (args.count("help")) {
      std::cout << opts << std::endl;
      return 1;
    }
    if (args.count("verbose")) {
      el::Loggers::setVerboseLevel(args["verbose"].as<int>());
    }
    args.notify();

    vertexai::tools::gencfg(args);

    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Caught unhandled exception: " << ex.what() << std::endl;
    auto stacktrace = boost::get_error_info<traced>(ex);
    if (stacktrace) {
      std::cerr << *stacktrace << std::endl;
    }
    return -1;
  } catch (...) {
    std::cerr << "Caught unhandled exception" << std::endl;
    return -1;
  }
}
