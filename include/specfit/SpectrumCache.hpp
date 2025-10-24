/* ===================================================================== *
 *  include/specfit/SpectrumCache.hpp   ––  bounded L-R-U cache (safe)
 * ===================================================================== */
#pragma once
#include "Spectrum.hpp"

#include <ankerl/unordered_dense.h>
#include <list>
#include <mutex>
#include <memory>
#include <optional>
#include <iostream>
#include <shared_mutex>

namespace specfit {

using SpectrumPtr = std::shared_ptr<const Spectrum>;

/*
 * Thread–safe bounded cache with   L-R-U eviction  and
 *                                   shared ownership.
 *
 * Public API:
 *   • SpectrumPtr try_get(key)
 *   • SpectrumPtr insert_if_absent(key, producer)
 *   • legacy bool fetch(key, Spectrum&)   (unchanged behaviour)
 *
 * A caller that needs the old  “const Spectrum&”  can simply dereference
 * the pointer it gets back:
 *
 *     const auto sp = cache.insert_if_absent(h, …);
 *     use(*sp);                      // operator*  gives a  const&.
 */
class SpectrumCache
{
public:
    static SpectrumCache& instance();

    /* ------------ zero-copy read (shared ownership) ---------------- */
    SpectrumPtr try_get(std::size_t hash) const;

    /* ------------ insert-or-get (preferred) ------------------------ */
    template<typename Producer>
    SpectrumPtr insert_if_absent(std::size_t hash, Producer&& make);

    /* ------------ legacy helpers ----------------------------------- */
    [[deprecated("Use try_get() or insert_if_absent()")]]
    bool fetch(std::size_t hash, Spectrum& out) const;
    void insert(std::size_t hash, Spectrum spec);

    /* ------------ house-keeping ------------------------------------ */
    void reserve(std::size_t n);
    void set_capacity(std::size_t n);
    void clear();

private:
    SpectrumCache() = default;

    /* ---------- internal L-R-U bookkeeping ------------------------- */
    using LruList = std::list<std::size_t>;                 // keys
    struct Node {
        SpectrumPtr       sp;       // shared ownership
        LruList::iterator lru_pos;  // position in the list
    };
    using Map = ankerl::unordered_dense::map<std::size_t, Node>;

    void touch_(typename Map::iterator it) const;   // header
    void evict_if_needed_();

    /* ---------- data members --------------------------------------- */
    mutable std::shared_mutex mtx_;
    mutable Map     cache_;
    mutable LruList lru_;
    std::size_t     max_entries_ = 2'048;
};

/* ===================================================================== *
 *  template implementation
 * ===================================================================== */
template<typename Producer>
SpectrumPtr SpectrumCache::insert_if_absent(std::size_t hash,
                                            Producer&&   make)
{
    {   
        std::unique_lock lk(mtx_);
        auto it = cache_.find(hash);
        if (it != cache_.end()) {
            touch_(it);
            return it->second.sp;          //  fast-path hit
        }
    }

    /* ---------- build spectrum outside any lock -------------------- */
    // NB: we build the *shared_ptr* directly
    SpectrumPtr new_sp = std::make_shared<Spectrum>(
                             std::forward<Producer>(make)() );

    /* ---------- second attempt / insertion ------------------------- */
    std::unique_lock lk(mtx_);
    auto it = cache_.find(hash);
    if (it != cache_.end()) {              // someone else inserted
        touch_(it);
        return it->second.sp;
    }

    auto lru_it = lru_.insert(lru_.begin(), hash);          // MRU front
    cache_.try_emplace(hash, Node{new_sp, lru_it});
    evict_if_needed_();
    return new_sp;
}

} // namespace specfit