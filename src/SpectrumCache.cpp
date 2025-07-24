#include "specfit/SpectrumCache.hpp"
#include <mutex>            // unique_lock

namespace specfit {

SpectrumCache& SpectrumCache::instance()
{
    static SpectrumCache inst;
    return inst;
}

/* ---------- write ---------------------------------------------------- */
void SpectrumCache::insert(std::size_t hash, Spectrum spec)
{
    // fast path: just delegate to generic helper
    (void) insert_if_absent(hash,
                            [&] { return std::move(spec); });
}

/* ---------- zero-copy read ------------------------------------------- */
const Spectrum* SpectrumCache::try_get(std::size_t hash) const
{
    std::shared_lock lk(mtx_);                      // shared/read lock
    auto it = cache_.find(hash);
    if (it == cache_.end()) return nullptr;
    return &it->second;                             // no copy
}

/* ---------- back-compat read (copy) ---------------------------------- */
bool SpectrumCache::fetch(std::size_t hash, Spectrum& out) const
{
    if (const auto* p = try_get(hash)) {
        out = *p;                       // copy for legacy callers
        return true;
    }
    return false;
}

} // namespace specfit