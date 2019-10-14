// Copyright 2018 Intel Corporation.
//
// This is the PlaidML base library interface, handling functionality common across the Vertex.AI libraries.

#pragma once

#if defined _WIN32 || defined __CYGWIN__
#ifdef VAI_DLL
#define VAI_API __declspec(dllexport)
#else
#define VAI_API __declspec(dllimport)
#endif
#elif __GNUC__ >= 4
#define VAI_API __attribute__((visibility("default")))
#else
#define VAI_API
#endif

#ifdef __cplusplus
#include <cstddef>
#include <memory>

extern "C" {
#else

#include <stdbool.h>
#include <stddef.h>

#endif  // __cplusplus

// Error handling.
//
// In the Vertex.AI C APIs, functions that can fail return either a pointer or a
// boolean; success is represented as a non-NULL or true value, and failure is
// represented as a NULL or false value.
//
// When a call fails, the callee is responsible for recording additional
// information in thread-local storage.  This information can be retrieved by
// calling vai_last_status().  Otherwise (if a call is successful), no
// guarantees are made; the callee may clobber existing thread-local error
// information, leaving it in an undefined state.
//
// Note that errors may occur asynchronously.  Asynchronous errors are reported
// to the various callback functions used to collect the results of
// asynchronous, operations, again via a NULL or false value.  In this case,
// the last error can be retrieved from thread-local-storage within the
// callback function.
//
// Failed calls always propagate errors, poisoning the state of any objects
// being updated.  Additionally, NULL inputs are valid for all calls, causing
// them to fail (on the assumption that a NULL input indicates an earlier
// out-of-memory condition).  Callers may take advantage of this by performing
// a sequence of calls without checking for errors, and then checking the last
// dependent call for errors; this obscures the exact call that produced an
// error, but in most cases the caller is more interested in the fact that the
// overall computation failed, and less interested in exactly which call
// failed.

// These are the various status codes the application may observe.
// Note that additional status codes may be added in subsquent releases.
//
// The set of status codes attempts to provide enough information for software components to determine the appropriate
// recovery action to take.  For diagnostics, the human-readable string associated with the status is typically more
// useful.
typedef enum {
  // A status representing "No error".
  VAI_STATUS_OK = 0,

  // Indicates that an asynchronous operations was cancelled.
  VAI_STATUS_CANCELLED = 1,

  // A generic catch-all error, used when an error condition must be signalled but
  // there is no appropriate API-level status code.
  VAI_STATUS_UNKNOWN = 2,

  // Indicates that at least one invalid argument was passed to a function.
  VAI_STATUS_INVALID_ARGUMENT = 3,

  // The operation deadline was exceeded.
  VAI_STATUS_DEADLINE_EXCEEDED = 4,

  // The requested object was not found.
  VAI_STATUS_NOT_FOUND = 5,

  // The requested object already exists.
  VAI_STATUS_ALREADY_EXISTS = 6,

  // The caller does not have permission to access a resource required by the operation.
  VAI_STATUS_PERMISSION_DENIED = 7,

  // A resource required by the operation is exhausted.  (For example, this is returned when the implementation is
  // unable to allocate sufficient memory.)
  VAI_STATUS_RESOURCE_EXHAUSTED = 8,

  // A precondition required by the operation is unmet.  (For example, this is returned when an object supplied to a
  // call is not in the correct state for the call to take place).
  VAI_STATUS_FAILED_PRECONDITION = 9,

  // A transactional operation was aborted by the system.  Generally, this is a transient condition.
  VAI_STATUS_ABORTED = 10,

  // A call parameter is out of the range accepted by the implementation.
  VAI_STATUS_OUT_OF_RANGE = 11,

  // The requested functionality is not implemented.
  VAI_STATUS_UNIMPLEMENTED = 12,

  // An internal error occurred.  Typically, this indicates that the implementation has detected that its internal state
  // is inconsistent.
  VAI_STATUS_INTERNAL = 13,

  // A resource required by the operation (such as a hardware device) is unavailable for use.
  VAI_STATUS_UNAVAILABLE = 14,

  // The system has lost data required by the operation, typically due to hardware failure.
  VAI_STATUS_DATA_LOSS = 15,

  // The caller is unauthenticated, but authenticated access is required to access some resource.
  VAI_STATUS_UNAUTHENTICATED = 16,
} vai_status;

// Returns the last status recorded in the current thread's thread-local storage,
// or VAI_STATUS_OK if no status has been recorded.
VAI_API vai_status vai_last_status();

// Resets the current thread's thread-local status storage to VAI_STATUS_OK.
VAI_API void vai_clear_status();

// Returns a NUL-terminated UTF-8 message describing the status of the call
// errors recorded by the current thread's thread-local storage.  If no error has
// been recorded, an empty string will be returned.
//
// The returned string will remain alive until vai_clear_error is called, another Vertex.AI
// call is made, or until the current thread exits.
//
// The error string may be dependent on the locale installed when the error occurred.
VAI_API const char* vai_last_status_str();

// Logger configuration.
typedef enum {
  VAI_LOG_SEVERITY_TRACE = 2,
  VAI_LOG_SEVERITY_DEBUG = 4,
  VAI_LOG_SEVERITY_FATAL = 8,
  VAI_LOG_SEVERITY_ERROR = 16,
  VAI_LOG_SEVERITY_WARNING = 32,
  VAI_LOG_SEVERITY_VERBOSE = 64,
  VAI_LOG_SEVERITY_INFO = 128,
} vai_log_severity;

// Sets the process-global logging callback.
VAI_API void vai_set_logger(void (*logger)(void*, vai_log_severity, const char*), void* arg);

// Dynamic feature detection.
//
// Invoking this API with an unknown / unsupported feature ID will return NULL.
// Invoking it with a supported feature ID will return a pointer to a static
// feature-specific value.
typedef enum {
  VAI_FEATURE_ID_RESERVED = 0,
} vai_feature_id;

VAI_API void* vai_query_feature(vai_feature_id id);

// A Vertex.AI context provides a scope for Vertex.AI library operations.  In
// particular, it provides an asynchronous execution context, allowing callers
// to correctly synchronize with asynchronous callbacks during shutdown.
//
// NULL semantically points to a valid, but cancelled, context.

#ifdef __cplusplus
struct vai_ctx;
#else
typedef struct vai_ctx vai_ctx;
#endif  // __cplusplus

// Allocate and returns a context, or returns NULL if the library
// cannot allocate sufficient memory.
VAI_API vai_ctx* vai_alloc_ctx();

// Frees a context.  After this call, the context should not be used for
// any subsequent calls.  Freeing a NULL context is a no-op.
//
// Freeing a context will block until pending asynchronous operations are complete.
VAI_API void vai_free_ctx(vai_ctx* ctx);

// Cancels outstanding asynchronous operations associated with the context
// (ensuring that callbacks are completed before returning to the caller), and
// causes future callbacks issued using the context to synchronously fail.
//
// Note that there is no call to go from "cancelled" back to "uncancelled".
//
// This does not block waiting for asynchronous operations to complete.
VAI_API void vai_cancel_ctx(vai_ctx* ctx);

// Sets the context to log events according to the specified
// configuration.  For instance, to point the context's event log at
// "eventlog.gz", use the JSON string:
//
//   "@type": "type.vertex.ai/vertexai.eventing.file.proto.EventLog",
//   "filename": "eventlog.gz"
//
// If the context already has an associated eventlog, that eventlog
// will be finalized and closed asynchronously, once all asynchronous
// activity using that eventlog has completed.
//
// A NULL config sets the context to not use the event logging
// subsystem for future calls.
VAI_API bool vai_set_eventlog(vai_ctx* ctx, const char* config);

// Gets the current value of a performance counter based on the name.
// If there is no performance counter with that name, returns -1.
VAI_API int64_t vai_get_perf_counter(const char* name);

// Sets the current value of a performance counter based on the name.
// If there is no performance counter with that name, no action is taken.
VAI_API void vai_set_perf_counter(const char* name, int64_t value);

// A PlaidML datatype indicates the type of data stored within a buffer, as
// observed by a program.
typedef enum {
  PLAIDML_DATA_INVALID = 0,
  PLAIDML_DATA_BOOLEAN = 0x02,
  PLAIDML_DATA_INT8 = 0x10,
  PLAIDML_DATA_INT16 = 0x11,
  PLAIDML_DATA_INT32 = 0x12,
  PLAIDML_DATA_INT64 = 0x13,
  PLAIDML_DATA_INT128 = 0x14,
  PLAIDML_DATA_UINT8 = 0x20,
  PLAIDML_DATA_UINT16 = 0x21,
  PLAIDML_DATA_UINT32 = 0x22,
  PLAIDML_DATA_UINT64 = 0x23,
  PLAIDML_DATA_FLOAT16 = 0x31,
  PLAIDML_DATA_FLOAT32 = 0x32,
  PLAIDML_DATA_FLOAT64 = 0x33,
  PLAIDML_DATA_BFLOAT16 = 0x38,
  PLAIDML_DATA_PRNG = 0x40,
} plaidml_datatype;

#ifdef __cplusplus
}  // extern "C"

namespace std {

template <>
struct default_delete<::vai_ctx> {
  void operator()(::vai_ctx* ctx) const noexcept { ::vai_free_ctx(ctx); }
};

}  // namespace std

#endif  // __cplusplus
