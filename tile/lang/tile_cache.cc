
#include "tile/lang/tile_cache.h"

#include "base/util/json_transfer.h"

namespace vertexai {
namespace tile {
namespace lang {

TileCache::TileCache(const std::string& filename, bool use_env) {
  std::string openname = filename;
  if (filename == "") {
    if (!use_env) {
      return;
    }
    char* name = std::getenv("PLAIDML_TILE_CACHE");
    if (!name) {
      return;
    }
    openname = std::string(name);
  }
  file_.exceptions(std::fstream::failbit | std::fstream::badbit);
  file_.open(openname, std::fstream::in | std::fstream::out | std::fstream::app);
  std::string line;
  file_.exceptions(std::fstream::badbit);
  while (std::getline(file_, line)) {
    Entry e = inline_json_deserialize<Entry>(line);
    AddEntry(e.key, e.subkey, e.value);
  }
  file_.clear();
  file_.exceptions(std::fstream::failbit | std::fstream::badbit);
}

TileCache* TileCache::Instance() {
  static TileCache instance("", true);
  return &instance;
}

void TileCache::AddEntry(const std::string& key, const DirectSettings& settings, const std::vector<uint64_t>& tile_size,
                         int64_t dur) {
  Entry e;
  e.key = key;
  e.subkey = Subkey(settings, tile_size);
  e.value = dur;
  AddEntry(key, e.subkey, dur);
  if (file_.is_open()) {
    std::string row = json_serialize(e);
    file_.write(row.data(), row.size());
    file_.flush();
  }
}

int64_t TileCache::GetDuration(const std::string& key, const DirectSettings& settings,
                               const std::vector<uint64_t>& tile_size) {
  auto it = cache_.find(key);
  if (it == cache_.end()) {
    return -1;
  }
  auto it2 = it->second.times.find(Subkey(settings, tile_size));
  if (it2 == it->second.times.end()) {
    return -1;
  }
  return it2->second;
}

void TileCache::AddEntry(const std::string& key, const Subkey& subkey, int64_t dur) {
  PerFC& p = cache_[key];
  p.times[subkey] = dur;
  if (p.times.size() == 1 || p.times[p.best] > dur) {
    p.best = subkey;
  }
}

TileCache::Subkey::Subkey(const DirectSettings& _settings, const std::vector<uint64_t>& _tile_size)
    : settings(_settings), tile_size(_tile_size) {}

bool TileCache::Subkey::operator<(const Subkey& rhs) const {
  if (settings.threads != rhs.settings.threads) {
    return settings.threads < rhs.settings.threads;
  }
  if (settings.use_global != rhs.settings.use_global) {
    return settings.use_global;
  }
  if (settings.mem_width != rhs.settings.mem_width) {
    return settings.mem_width < rhs.settings.mem_width;
  }
  return tile_size < rhs.tile_size;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
