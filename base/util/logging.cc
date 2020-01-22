#include <gflags/gflags.h>

#include <string>

#include "base/util/logging.h"

INITIALIZE_EASYLOGGINGPP;

DEFINE_bool(logtofile, false, "enable logfile output");
DEFINE_int32(v, 0, "enable verbose (DEBUG) logging");
DEFINE_string(vmodule, "", "enable verbose (DEBUG) logging");
DEFINE_string(logconf, "", "enable logging configuration from file");

namespace {
#if ELPP_OS_WINDOWS
const char logDirPrefix[] = "logs\\";  // NOLINT
#else
const char logDirPrefix[] = "logs/";  // NOLINT
#endif
}  // namespace

namespace vertexai {

el::Configurations LogConfigurationFromFlags(const std::string& app_name) {
  el::Configurations conf;
  if (FLAGS_logconf.empty()) {
    conf.setToDefault();
  } else {
    conf = el::Configurations(FLAGS_logconf.c_str());
  }
  if (!FLAGS_logtofile) {
    conf.set(el::Level::Global, el::ConfigurationType::ToFile, "false");
  } else {
    conf.set(el::Level::Global, el::ConfigurationType::Filename, std::string(logDirPrefix) + app_name + ".log");
  }
  if (!FLAGS_v) {
    conf.set(el::Level::Debug, el::ConfigurationType::Enabled, "false");
  } else {
    el::Loggers::setVerboseLevel(FLAGS_v);
  }
  if (!FLAGS_vmodule.empty()) {
    el::Loggers::setVModules(FLAGS_vmodule.c_str());
  }

  return conf;
}
}  // namespace vertexai
