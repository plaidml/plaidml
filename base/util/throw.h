#pragma once

#include <boost/exception/all.hpp>
#include <boost/stacktrace.hpp>

using traced = boost::error_info<struct tag_stacktrace, boost::stacktrace::stacktrace>;

template <typename E>
void throw_with_trace(const E& ex) {
  throw boost::enable_error_info(ex) << traced(boost::stacktrace::stacktrace());
}
