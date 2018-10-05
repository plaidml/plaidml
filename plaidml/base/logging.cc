// Copyright 2018 Intel Corporation.

#include <mutex>

#include "base/util/logging.h"
#include "plaidml/base/base.h"

namespace vertexai {

class ExternalLogger final : public el::LogDispatchCallback {
 public:
  typedef void (*Callback)(void*, vai_log_severity, const char*);

  static void SetLoggerCallback(Callback logger, void* arg);

  void handle(const el::LogDispatchData* data) final;

 private:
  static std::mutex g_mu;
  static Callback g_logger;
  static void* g_arg;
  static el::Configurations g_previous_config;
};

std::mutex ExternalLogger::g_mu;
ExternalLogger::Callback ExternalLogger::g_logger = nullptr;
void* ExternalLogger::g_arg = nullptr;
el::Configurations ExternalLogger::g_previous_config;

void ExternalLogger::SetLoggerCallback(Callback logger, void* arg) {
  std::unique_lock<std::mutex> lock{g_mu};
  if (logger && !g_logger) {
    el::Helpers::installLogDispatchCallback<ExternalLogger>("external");
    el::Configurations config;
    auto el_logger = el::Loggers::getLogger("default");
    config = *el_logger->configurations();
    g_previous_config = config;
    config.setGlobally(el::ConfigurationType::ToStandardOutput, "false");
    el_logger->configure(config);
  } else if (!logger && g_logger) {
    el::Loggers::reconfigureLogger("default", g_previous_config);
    el::Helpers::uninstallLogDispatchCallback<ExternalLogger>("external");
  }

  g_logger = logger;
  g_arg = arg;
}

void ExternalLogger::handle(const el::LogDispatchData* data) {
  std::unique_lock<std::mutex> lock{g_mu};
  if (!g_logger) {
    // N.B. This should never happen, but we'll deal with it anyway,
    // just in case.
    return;
  }

  // TODO: Consider taking the dispatch action into account; also
  // consider including file/line/func/verbosity/logger name/&c.
  // Perhaps we should be passing back some sort of extensible struct?

  g_logger(g_arg, static_cast<vai_log_severity>(data->logMessage()->level()), data->logMessage()->message().c_str());
}

}  // namespace vertexai

extern "C" VAI_API void vai_internal_set_vlog(size_t num) { el::Loggers::setVerboseLevel(num); }

extern "C" void vai_set_logger(vertexai::ExternalLogger::Callback logger, void* arg) {
  vertexai::ExternalLogger::SetLoggerCallback(logger, arg);
}
