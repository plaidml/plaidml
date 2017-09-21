#pragma once

#include <string>

#ifdef _GNUC
#define ATTR_PRINTF(format, params) __attribute__((format(printf, format, params)))
#else
#define ATTR_PRINTF(format, params)
#endif

std::string printstring(const char* format, ...) ATTR_PRINTF(1, 2);
