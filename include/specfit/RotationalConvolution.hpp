// RotationalConvolution.hpp
#pragma once
#include "Types.hpp"
#include <unordered_map>
#include <mutex>
#include <memory>

namespace specfit {

// Cache for rotational kernels
class RotationalKernelCache {
public:
    struct KernelData {
        Vector vel_shift;   // Velocity shifts in km/s
        Vector vel_kernel;  // Normalized kernel weights
        double vsini_kms;
        double epsilon;
        int n_kernel;
    };
    
    using KernelPtr = std::shared_ptr<const KernelData>;
    
    static RotationalKernelCache& instance() {
        static RotationalKernelCache cache;
        return cache;
    }
    
    KernelPtr get_or_compute(double vsini_kms, double epsilon, int n_kernel);
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
    }
    
    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }
    
private:
    struct Key {
        double vsini_kms;
        double epsilon;
        int n_kernel;
        
        bool operator==(const Key& other) const {
            return std::abs(vsini_kms - other.vsini_kms) < 1e-6 &&
                   std::abs(epsilon - other.epsilon) < 1e-6 &&
                   n_kernel == other.n_kernel;
        }
    };
    
    struct KeyHash {
        std::size_t operator()(const Key& k) const {
            std::size_t h1 = std::hash<double>{}(std::round(k.vsini_kms * 1e6) / 1e6);
            std::size_t h2 = std::hash<double>{}(std::round(k.epsilon * 1e6) / 1e6);
            std::size_t h3 = std::hash<int>{}(k.n_kernel);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
    
    mutable std::mutex mutex_;
    std::unordered_map<Key, KernelPtr, KeyHash> cache_;
    
    RotationalKernelCache() = default;
    RotationalKernelCache(const RotationalKernelCache&) = delete;
    RotationalKernelCache& operator=(const RotationalKernelCache&) = delete;
};

// Main functions
Vector rotational_broaden(const Vector& lam,
                         const Vector& flux,
                         double vsini_kms,
                         double epsilon = 0.6,
                         int n_kernel = 81);

Vector rotational_broaden_nocache(const Vector& lam,
                                  const Vector& flux,
                                  double vsini_kms,
                                  double epsilon = 0.6,
                                  int n_kernel = 81);

} // namespace specfit