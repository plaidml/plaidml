#pragma once

#include <istream>
#include <string>

#define JSON_HAS_INT64
#include "json/json.h"
#include "json/reader.h"

#include "base/util/transfer_object.h"

namespace vertexai {

void throw_bad_type(const Json::ValueType& found_type, const Json::ValueType& expected_type);

template <class T>
Json::Value json_wrap(T& val);  // NOLINT

template <class ST, class T>
struct json_wrap_impl {};

#define WRAP_PRIMITIVE(ctype, jtype)                                                           \
  template <class T>                                                                           \
  struct json_wrap_impl<ctype, T> {                                                            \
    static Json::Value wrap(T& val) { return Json::Value((jtype)transfer_info<T>::get(val)); } \
  };

WRAP_PRIMITIVE(int64_t, Json::Int64)
WRAP_PRIMITIVE(uint64_t, Json::UInt64)
WRAP_PRIMITIVE(double, double)
WRAP_PRIMITIVE(bool, bool)
WRAP_PRIMITIVE(std::string, std::string)

template <class T>
struct json_wrap_impl<transfer_type_null, T> {
  static Json::Value wrap(T& val) {  // NOLINT
    return Json::Value();
  }
};

class json_serialize_context {
 public:
  void set_version(size_t version) {
    if (version != 0) {
      m_obj["_ver"] = Json::Value::UInt64(version);
    }
    m_version = version;
  }

  bool is_serialize() { return true; }
  bool human_readable() { return true; }
  bool has_field(const std::string& name, size_t tag) { return true; }
  bool is_null(const std::string& name, size_t tag) { return false; }
  size_t get_version() { return m_version; }

  template <class T>
  void transfer_field(const std::string& name, size_t tag, T& obj) {  // NOLINT
    m_obj[name] = json_wrap(obj);
  }
  const Json::Value& get_result() { return m_obj; }

 private:
  size_t m_version = 0;
  Json::Value m_obj;
};

template <>
struct json_wrap_impl<transfer_type_object, Json::Value> {
  static Json::Value wrap(Json::Value& val) {  // NOLINT
    return val;
  }
};

template <class T>
struct json_wrap_impl<transfer_type_object, T> {
  static Json::Value wrap(T& val) {  // NOLINT
    json_serialize_context ctx;
    transfer_info<T>::object_transfer(ctx, val);
    return ctx.get_result();
  }
};

template <class T>
struct json_wrap_impl<transfer_type_array, T> {
  static Json::Value wrap(T& val) {  // NOLINT
    typedef typename transfer_info<T>::iterator iterator;
    iterator itEnd = transfer_info<T>::end(val);
    Json::Value r;
    for (iterator it = transfer_info<T>::begin(val); it != itEnd; ++it) {
      r.append(json_wrap(*it));
    }
    return r;
  }
};

/*
template <class T>
struct json_wrap_impl<transfer_type_tuple, T>
{
    static void apply_tuple(Json::Value& r, const boost::tuples::null_type& nothing)
    {}

    template <class Head, class Tail>
    static void apply_tuple(Json::Value& r, boost::tuples::cons<Head, Tail>& x)
    {
        r.push_back(json_wrap(x.get_head()));
        apply_tuple(r, x.get_tail());
    }

    static Json::Value wrap(T& val)
    {
        Json::mArray r;
        typename transfer_info<T>::tuple_type tuple = transfer_info<T>::as_boost_tuple(val);
        apply_tuple(r, tuple);
        return r;
    }
};
*/

template <class T>
struct json_wrap_impl<transfer_type_map_object, T> {
  static Json::Value wrap(T& val) {  // NOLINT
    Json::Value r;
    typename T::iterator itEnd = val.end();
    for (typename T::iterator it = val.begin(); it != itEnd; ++it) {
      r[it->first] = json_wrap(it->second);
    }
    return r;
  }
};

template <class T>
Json::Value json_wrap(T& val) {  // NOLINT
  typedef typename transfer_info<T>::type sub_type;
  return json_wrap_impl<sub_type, T>::wrap(val);
}

template <class T>
std::string json_serialize(const T& obj, bool pretty = false) {
  Json::Value json = json_wrap(const_cast<T&>(obj));
  if (pretty) {
    Json::StyledWriter w;
    return w.write(json);
  } else {
    Json::FastWriter w;
    return w.write(json);
  }
}

template <class T>
void json_unwrap(T& val, const Json::Value& json);  // NOLINT

template <class ST, class T>
struct json_unwrap_impl {};

#define UNWRAP_PRIMITIVE(cname, jname)                                    \
  template <class T>                                                      \
  struct json_unwrap_impl<cname, T> {                                     \
    static void unwrap(T& val, const Json::Value& json) {                 \
      if (!json.is##jname()) {                                            \
        throw deserialization_error("Invalid type, looking for " #cname); \
      }                                                                   \
      transfer_info<T>::put(val, json.as##jname());                       \
    }                                                                     \
  };

UNWRAP_PRIMITIVE(int64_t, Int64)
UNWRAP_PRIMITIVE(uint64_t, UInt64)
UNWRAP_PRIMITIVE(double, Double)
UNWRAP_PRIMITIVE(bool, Bool)
UNWRAP_PRIMITIVE(std::string, String)

class json_deserialize_context {
 public:
  explicit json_deserialize_context(const Json::Value& obj) : m_obj(obj) {}

  void set_version(size_t version) {}
  bool is_serialize() { return false; }
  bool human_readable() { return true; }

  bool has_field(const std::string& name, size_t tag) { return m_obj.isMember(name); }

  bool is_null(const std::string& name, size_t tag) { return m_obj[name].type() == Json::nullValue; }

  size_t get_version() {
    if (!has_field("_ver", 0)) {
      return 0;
    }
    uint64_t version;
    json_unwrap(version, m_obj["_ver"]);
    return version;
  }

  template <class T>
  void transfer_field(const std::string& name, size_t tag, T& obj) {  // NOLINT
    const Json::Value& val = m_obj[name];
    if (val.isNull()) {
      throw deserialization_error("Null field or missing field: " + name);
    }
    json_unwrap(obj, val);
  }

 private:
  const Json::Value& m_obj;
};

template <class T>
struct json_unwrap_impl<transfer_type_object, T> {
  static void unwrap(T& val, const Json::Value& json) {  // NOLINT
    if (json.type() != Json::objectValue) {
      throw_bad_type(json.type(), Json::objectValue);
    }
    json_deserialize_context ctx(json);
    transfer_info<T>::object_transfer(ctx, val);
  }
};

template <>
struct json_unwrap_impl<transfer_type_object, Json::Value> {
  static void unwrap(Json::Value& val, const Json::Value& json) {  // NOLINT
    val = json;
  }
};

template <class T>
struct json_unwrap_impl<transfer_type_array, T> {
  static void unwrap(T& val, const Json::Value& json) {  // NOLINT
    typedef typename transfer_info<T>::value_type value_type;

    if (json.type() != Json::arrayValue) {
      throw_bad_type(json.type(), Json::arrayValue);
    }

    val = T();

    for (int i = 0; i < json.size(); i++) {
      value_type r;
      json_unwrap(r, json[i]);
      transfer_info<T>::push_back(val, r);
    }
  }
};

/*
template <class T>
struct json_unwrap_impl<transfer_type_tuple, T>
{
    static void apply_tuple(const Json::mArray& r, size_t i, const boost::tuples::null_type& nothing)
    {}

    template <class Head, class Tail>
    static void apply_tuple(const Json::mArray& r, size_t i, boost::tuples::cons<Head, Tail>& x)
    {
        json_unwrap(x.get_head(), r[i]);
        apply_tuple(r, i+1, x.get_tail());
    }

    static void unwrap(T& val, const Json::Value& json)
    {
        if (json.type() != Json::array_type) {
            throw_bad_type(json.type(), Json::array_type);
        }
        const Json::mArray& arr = json.get_array();
        typedef typename transfer_info<T>::tuple_type tuple_type;
        if (arr.size() != (size_t) boost::tuples::length<tuple_type>::value) {
            throw deserialization_error("Invalid number of tuple elements");
        }
        tuple_type out_tuple;
        apply_tuple(arr, 0, out_tuple);
        transfer_info<T>::from_boost_tuple(val, out_tuple);
    }
};
*/

template <class T>
struct json_unwrap_impl<transfer_type_map_object, T> {
  static void unwrap(T& val, const Json::Value& json) {  // NOLINT
    typedef typename transfer_info<T>::value_type value_type;

    if (json.type() != Json::objectValue) {
      throw_bad_type(json.type(), Json::objectValue);
    }

    val = T();

    for (Json::Value::const_iterator it = json.begin(); it != json.end(); ++it) {
      value_type item_value;
      json_unwrap(item_value, *it);
      val[it.key()] = item_value;
    }
  }
};

template <class T>
void json_unwrap(T& val, const Json::Value& json) {  // NOLINT
  typedef typename transfer_info<T>::type sub_type;
  json_unwrap_impl<sub_type, T>::unwrap(val, json);
}

template <class T>
void json_deserialize(T& out, const std::string& str) {  // NOLINT
  Json::Reader reader;
  Json::Value json;
  bool r = reader.parse(str, json);
  if (!r) {
    throw deserialization_error(reader.getFormattedErrorMessages());
  }
  json_unwrap(out, json);
}

template <class T>
T inline_json_deserialize(const std::string& str) {
  T result;
  json_deserialize(result, str);
  return result;
}

}  // End namespace vertexai
