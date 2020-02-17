// Copyright 2020 Intel Corporation

#pragma once

struct UnrankedMemRefType {
  int64_t rank;
  void *descriptor;
};

/// StridedMemRef descriptor type with static rank.
template <typename T, int N>
struct StridedMemRefType {
  T *basePtr;
  T *data;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};
