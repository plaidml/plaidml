#include "json_transfer.h"

namespace vertexai {

static const std::map<Json::ValueType, std::string> g_type_to_str{
    {Json::objectValue, "object"}, {Json::arrayValue, "array"}, {Json::stringValue, "string"},
    {Json::booleanValue, "bool"},  {Json::intValue, "int"},     {Json::realValue, "real"},
    {Json::nullValue, "null"}};

std::string exception_msg(const Json::ValueType& t) { return printstring("unknown json type with enum %d", t); }

void throw_bad_type(const Json::ValueType& found_type, const Json::ValueType& expected_type) {
  auto found_it = g_type_to_str.find(found_type);
  auto expected_it = g_type_to_str.find(expected_type);
  if (found_it == g_type_to_str.end()) {
    throw deserialization_error(exception_msg(found_type));
  }
  if (expected_it == g_type_to_str.end()) {
    throw deserialization_error(exception_msg(expected_type));
  }

  std::string found_type_as_str = found_it->second;
  std::string expected_type_as_str = expected_it->second;
  throw deserialization_error("Json is of type " + found_type_as_str + ", not an " + expected_type_as_str);
}

}  // namespace vertexai
