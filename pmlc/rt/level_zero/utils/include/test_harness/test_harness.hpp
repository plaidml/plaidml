/*
 *
 * Copyright (C) 2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */

#ifndef level_zero_tests_ZE_TEST_HARNESS_HPP
#define level_zero_tests_ZE_TEST_HARNESS_HPP

#include <assert.h>
#define EXPECT_EQ(x, y) assert((x) == (y))
#define EXPECT_NE(x, y) assert((x) != (y))
#define EXPECT_GT(x, y) assert((x) > (y))
#define EXPECT_TRUE(x) assert((x))

#define ASSERT_EQ(x, y) EXPECT_EQ(x, y)

//#include "test_harness_driver.hpp"
#include "test_harness_cmdlist.hpp"
#include "test_harness_cmdqueue.hpp"
#include "test_harness_device.hpp"
//#include "test_harness_fence.hpp"
//#include "test_harness_event.hpp"
#include "test_harness_memory.hpp"
//#include "test_harness_image.hpp"
#include "test_harness_module.hpp"
//#include "test_harness_sampler.hpp"
//#include "test_harness_ocl_interop.hpp"
//#include "test_harness_driver_info.hpp"
//#include "../../tools/include/test_harness_api_tracing.hpp"
//#include "../../tools/include/test_harness_api_ltracing.hpp"
//#include "../../tools/sysman/include/test_harness_sysman.hpp"
//#include "../../tools/include/test_harness_metric.hpp"

#endif
