// Copyright 2018 Intel Corporation.

#pragma once

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include <utility>

namespace vertexai {
namespace compat {

template <class T>
using remove_cv_t = typename std::remove_cv<T>::type;

template <class T>
using remove_reference_t = typename std::remove_reference<T>::type;

}  // namespace compat
}  // namespace vertexai
