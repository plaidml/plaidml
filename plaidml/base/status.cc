// Copyright 2018 Intel Corporation.

#include "plaidml/base/status.h"

#include <string>

#include "base/util/error.h"
#include "base/util/logging.h"
#include "plaidml/base/status_strings.h"

namespace {

// TODO: T451
#if TARGET_OS_IPHONE == 1
vai_status last_status = VAI_STATUS_OK;
std::string last_status_str;  // NOLINT
#else                         // TARGET_OS_IPHONE == 1
thread_local vai_status last_status = VAI_STATUS_OK;
thread_local std::string last_status_str;
#endif                        // TARGET_OS_IPHONE == 1 ... else

}  // namespace

namespace vertexai {

namespace {
class ErrToCode final : public error::ErrorVisitor {
 public:
  void Visit(const error::Cancelled& err) noexcept final { status_ = VAI_STATUS_CANCELLED; }
  void Visit(const error::Unknown& err) noexcept final { status_ = VAI_STATUS_UNKNOWN; }
  void Visit(const error::InvalidArgument& err) noexcept final { status_ = VAI_STATUS_INVALID_ARGUMENT; }
  void Visit(const error::DeadlineExceeded& err) noexcept final { status_ = VAI_STATUS_DEADLINE_EXCEEDED; }
  void Visit(const error::NotFound& err) noexcept final { status_ = VAI_STATUS_NOT_FOUND; }
  void Visit(const error::AlreadyExists& err) noexcept final { status_ = VAI_STATUS_ALREADY_EXISTS; }
  void Visit(const error::PermissionDenied& err) noexcept final { status_ = VAI_STATUS_PERMISSION_DENIED; }
  void Visit(const error::Unauthenticated& err) noexcept final { status_ = VAI_STATUS_UNAUTHENTICATED; }
  void Visit(const error::ResourceExhausted& err) noexcept final { status_ = VAI_STATUS_RESOURCE_EXHAUSTED; }
  void Visit(const error::FailedPrecondition& err) noexcept final { status_ = VAI_STATUS_FAILED_PRECONDITION; }
  void Visit(const error::Aborted& err) noexcept final { status_ = VAI_STATUS_ABORTED; }
  void Visit(const error::OutOfRange& err) noexcept final { status_ = VAI_STATUS_OUT_OF_RANGE; }
  void Visit(const error::Unimplemented& err) noexcept final { status_ = VAI_STATUS_UNIMPLEMENTED; }
  void Visit(const error::Internal& err) noexcept final { status_ = VAI_STATUS_INTERNAL; }
  void Visit(const error::Unavailable& err) noexcept final { status_ = VAI_STATUS_UNAVAILABLE; }
  void Visit(const error::DataLoss& err) noexcept final { status_ = VAI_STATUS_DATA_LOSS; }

  vai_status status() const noexcept { return status_; }

 private:
  vai_status status_ = VAI_STATUS_OK;
};
}  // namespace

void SetLastStatus(vai_status status, const char* str) noexcept {
  if (!str || !*str) {
    status = VAI_STATUS_INTERNAL;
    str = status_strings::kInternal;
  }
  last_status = status;
  try {
    last_status_str = str;
    if (status != VAI_STATUS_OK) {
      IVLOG(1, "ERROR: " << str);
    }
  } catch (...) {
    // Replacing the status code with VAI_STATUS_RESOURCE_EXHAUSTED lets
    // vai_last_status_str() return a generic out-of-resources message.
    last_status = VAI_STATUS_RESOURCE_EXHAUSTED;
    last_status_str.clear();
  }
}

void SetLastException(std::exception_ptr ep) noexcept {
  try {
    std::rethrow_exception(ep);
  } catch (const error::Error& e) {
    ErrToCode coder;
    e.Accept(&coder);
    SetLastStatus(coder.status(), e.what());
  } catch (const std::bad_alloc&) {
    SetLastStatus(VAI_STATUS_RESOURCE_EXHAUSTED, status_strings::kOom);
  } catch (const std::exception& e) {
    SetLastStatus(VAI_STATUS_UNKNOWN, e.what());
  } catch (...) {
    SetLastStatus(VAI_STATUS_INTERNAL, "The exception chain appears to be corrupt");
  }
}

void SetLastOOM() noexcept { SetLastStatus(VAI_STATUS_RESOURCE_EXHAUSTED, status_strings::kOom); }

}  // namespace vertexai

// vai_status
extern "C" vai_status vai_last_status() { return last_status; }

extern "C" void vai_clear_status() { vertexai::SetLastStatus(VAI_STATUS_OK, vertexai::status_strings::kOk); }

extern "C" const char* vai_last_status_str() {
  if (last_status == VAI_STATUS_RESOURCE_EXHAUSTED && !last_status_str.length()) {
    return vertexai::status_strings::kOom;
  }
  return last_status_str.c_str();
}
