// Copyright 2018 Intel Corporation.

#pragma once

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

#include <memory>
#include <utility>

namespace vertexai {
namespace compat {

template <class T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <class T>
using remove_cv_t = typename std::remove_cv<T>::type;

template <class T>
using remove_reference_t = typename std::remove_reference<T>::type;

}  // namespace compat
}  // namespace vertexai
