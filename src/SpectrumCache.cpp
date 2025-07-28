/* ===================================================================== *
 *  src/SpectrumCache.cpp
 * ===================================================================== */
#include "specfit/SpectrumCache.hpp"

namespace specfit {

/* -------- singleton -------------------------------------------------- */
SpectrumCache& SpectrumCache::instance()
{
    static SpectrumCache inst;
    return inst;
}

/* -------- simple helpers -------------------------------------------- */
void SpectrumCache::reserve(std::size_t n)
{
    std::unique_lock lk(mtx_);
    cache_.reserve(n);
}
void SpectrumCache::set_capacity(std::size_t n)
{
    std::unique_lock lk(mtx_);
    max_entries_ = (n == 0) ? 1 : n;
    evict_if_needed_();
}
void SpectrumCache::clear()
{
    std::unique_lock lk(mtx_);
    cache_.clear();
    lru_.clear();
}

/* -------- try_get ---------------------------------------------------- */
SpectrumPtr SpectrumCache::try_get(std::size_t hash) const
{
    std::unique_lock lk(mtx_);
    auto it = cache_.find(hash);
    if (it == cache_.end()) return nullptr;
    touch_(it);                      // update LRU even on read
    return it->second.sp;
}

/* -------- back-compat helpers --------------------------------------- */
void SpectrumCache::insert(std::size_t hash, Spectrum spec)
{
    insert_if_absent(hash, [&]{ return std::move(spec); });
}
bool SpectrumCache::fetch(std::size_t hash, Spectrum& out) const
{
    if (auto sp = try_get(hash)) {
        out = *sp;                   // deep copy for legacy callers
        return true;
    }
    return false;
}

/* ===================================================================== *
 *            internal L-R-U helpers (private)
 * ===================================================================== */
void SpectrumCache::touch_(typename Map::iterator it) const
{
    lru_.splice(lru_.begin(), lru_, it->second.lru_pos);
}
void SpectrumCache::evict_if_needed_()
{
    while (cache_.size() > max_entries_) {
        std::size_t victim = lru_.back();
        lru_.pop_back();
        cache_.erase(victim);        // shared_ptr keeps data alive
    }
}

} // namespace specfit