// Copyright 2018 Intel Corporation.

#include "plaidml/base/context.h"

#include <memory>
#include <utility>

#include "base/config/config.h"
#include "base/util/any_factory_map.h"
#include "plaidml/base/status.h"

namespace context = vertexai::context;

extern "C" vai_ctx* vai_alloc_ctx() {
  try {
    return new vai_ctx{
        context::Activity{context::Context().set_gate(std::make_shared<context::Gate>()), "vertexai::TopLevel", true}};
  } catch (...) {
    vertexai::SetLastOOM();
    return nullptr;
  }
}

extern "C" void vai_free_ctx(vai_ctx* ctx) {
  if (ctx) {
    ctx->activity.ctx().gate()->Close().wait();
    ctx->activity.mutable_ctx()->set_is_logging_events(false);
    if (ctx->activity.ctx().eventlog()) {
      ctx->activity.ctx().eventlog()->FlushAndClose();
    }
  }
  delete ctx;
}

extern "C" void vai_cancel_ctx(vai_ctx* ctx) {
  if (!ctx) {
    return;
  }
  ctx->activity.ctx().gate()->Close();
}

extern "C" bool vai_set_eventlog(vai_ctx* ctx, const char* config) {
  if (!ctx) {
    vertexai::SetLastOOM();
    return false;
  }
  if (config) {
    try {
      auto pconfig = vertexai::ParseConfig<google::protobuf::Any>(config);
      auto eventlog =
          vertexai::AnyFactoryMap<vertexai::context::EventLog>::Instance()->MakeInstance(ctx->activity.ctx(), pconfig);
      ctx->activity.mutable_ctx()->set_eventlog(std::move(eventlog));
      ctx->activity.mutable_ctx()->set_is_logging_events(true);
    } catch (...) {
      vertexai::SetLastException(std::current_exception());
      return false;
    }
  } else {
    ctx->activity.mutable_ctx()->set_is_logging_events(false);
    ctx->activity.mutable_ctx()->set_eventlog(std::shared_ptr<vertexai::context::EventLog>());
  }
  return true;
}
