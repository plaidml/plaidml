
#pragma once

#include <fstream>
#include <limits>
#include <vector>

#include "base/util/transfer_object.h"
#include "tile/lang/generate.h"

namespace vertexai {
namespace tile {
namespace lang {

class TileCache {
 public:
  // Construct a cache, if given a filename, use that for storage
  TileCache(const std::string& filename = "", bool use_env = false);
  // Get the 'singlton' instance, loads for PLAIDML_TILE_CACHE if set
  static TileCache* Instance();
  // Add a new entry with a duration
  void AddEntry(const std::string& key, const DirectSettings& settings, const std::vector<uint64_t>& tile_size,
                int64_t dur);
  // Checks for an exact matching entry (to skip tile scan for repeats), or -1 if not found
  int64_t GetDuration(const std::string& key, const DirectSettings& settings, const std::vector<uint64_t>& tile_size);

 private:
  struct Subkey {
    Subkey() = default;  // For deserialization
    Subkey(const DirectSettings& settings, const std::vector<uint64_t>& tile_size);
    bool operator<(const Subkey& rhs) const;

    DirectSettings settings;
    std::vector<uint64_t> tile_size;

    TRANSFER_OBJECT {
      VERSION(0);
      FIELD(settings);
      FIELD(tile_size);
    }
  };

  struct Entry {
    std::string key;
    TileCache::Subkey subkey;
    int64_t value;

    TRANSFER_OBJECT {
      VERSION(0);
      FIELD(key);
      FIELD(subkey);
      FIELD(value);
    }
  };

  struct PerFC {
    Subkey best;
    std::map<Subkey, int64_t> times;
  };

  void AddEntry(const std::string& key, const Subkey& subkey, int64_t dur);

  std::map<const std::string, PerFC> cache_;

  std::fstream file_;
};

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
