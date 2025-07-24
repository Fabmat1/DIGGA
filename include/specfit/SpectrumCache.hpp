#pragma once
#include "Spectrum.hpp"

#include <unordered_map>
#include <shared_mutex>     // shared_mutex / shared_lock
#include <optional>
#include <functional>       // reference_wrapper
#include <type_traits>
#include <mutex>
#include <cstddef>          // std::size_t

#include <ankerl/unordered_dense.h>      // fast flat hash map
namespace specfit {

/*
 * Thread-safe cache.
 *  • Reads take a shared lock and, if you want, return a const reference
 *    (zero-copy).  Back-compat bool-returning API kept, too.
 *  • Writes take an exclusive lock.
 */
class SpectrumCache
{
public:
    static SpectrumCache& instance();

    /* New zero-copy API */
    const Spectrum* try_get(std::size_t hash) const;
    
    /*
     * Insert a Spectrum **only if the key is absent** and return a reference
     * to the cached object.  The `Producer` is called *outside* any lock.
     *
     *   const Spectrum& sp = cache.insert_if_absent(h, [&]{
     *         return read_fits(path);      // heavy work done once
     *   });
     */
    template<typename Producer>
    const Spectrum& insert_if_absent(std::size_t hash, Producer&& make);
    /* Backward-compatible API used by existing code: copies into 'out'. */
    [[deprecated("Use try_get() or insert_if_absent()")]]
    bool fetch(std::size_t hash, Spectrum& out) const;

    void insert(std::size_t hash, Spectrum spec);

    /* Optional: pre-allocate */
    void reserve(std::size_t n) { cache_.reserve(n); }

private:
    SpectrumCache() = default;

    mutable std::shared_mutex                    mtx_;
    using Map = ankerl::unordered_dense::map<std::size_t, Spectrum>;
    Map                                          cache_;
};

/* ===================================================================== *
 *  template impl. must stay in header
 * ===================================================================== */

template<typename Producer>
const Spectrum& SpectrumCache::insert_if_absent(std::size_t hash,
                                                Producer&&   make)
{
    // ---- 1st probe under shared lock --------------------------------
    {
        std::shared_lock rlk(mtx_);
        auto it = cache_.find(hash);
        if (it != cache_.end()) return it->second;
    }

    // create outside any lock
    Spectrum tmp = std::forward<Producer>(make)();

    // ---- 2nd probe under exclusive lock -----------------------------
    std::unique_lock wlk(mtx_);
    auto [it, inserted] = cache_.try_emplace(hash, std::move(tmp));
    return it->second;
}

} // namespace specfit