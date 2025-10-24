#include "specfit/Resolution.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>
#include <tuple>
#include <list>

#ifdef _OPENMP
  #include <omp.h>
#endif

namespace specfit {

using Vector = Eigen::VectorXd;

namespace {
constexpr double SIGMA_FROM_FWHM = 1.0 / (2.0 * std::sqrt(2.0 * std::log(2.0)));
constexpr double KERNEL_RADIUS   = 5.0;

// Structure to hold precomputed weights for one output point
struct WeightSegment {
    std::size_t jStart;
    std::size_t jEnd;
    std::vector<double> weights;  // Pre-normalized weights including dLam
};

// Cache key: wavelength grid signature + resolution parameters
struct CacheKey {
    double lamStart;
    double lamEnd;
    double avgDelta;
    std::size_t nPoints;
    double resOffset;
    double resSlope;
    
    bool operator==(const CacheKey& other) const {
        return std::abs(lamStart - other.lamStart) < 1e-10 &&
               std::abs(lamEnd - other.lamEnd) < 1e-10 &&
               std::abs(avgDelta - other.avgDelta) < 1e-10 &&
               nPoints == other.nPoints &&
               std::abs(resOffset - other.resOffset) < 1e-10 &&
               std::abs(resSlope - other.resSlope) < 1e-10;
    }
};

// Hash function for cache key
struct CacheKeyHash {
    std::size_t operator()(const CacheKey& k) const {
        auto h1 = std::hash<double>{}(k.lamStart);
        auto h2 = std::hash<double>{}(k.lamEnd);
        auto h3 = std::hash<double>{}(k.avgDelta);
        auto h4 = std::hash<std::size_t>{}(k.nPoints);
        auto h5 = std::hash<double>{}(k.resOffset);
        auto h6 = std::hash<double>{}(k.resSlope);
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5);
    }
};

// Global cache (thread-safe for reads after initial computation)
static std::unordered_map<CacheKey, std::vector<WeightSegment>, CacheKeyHash> g_weightCache;
static std::list<CacheKey> g_lruList;
static std::mutex g_cacheMutex;
static constexpr std::size_t MAX_CACHE_SIZE = 25;

// Moves key to front (most recently used)
static void touch_key(const CacheKey& key) {
    // Remove any existing occurrence
    g_lruList.remove(key);
    g_lruList.push_front(key);
}

// Inserts key and evicts if needed
static void insert_cache_entry(const CacheKey& key, std::vector<WeightSegment>&& value) {
    g_weightCache.emplace(key, std::move(value));
    g_lruList.push_front(key);
    
    if (g_weightCache.size() > MAX_CACHE_SIZE) {
        const CacheKey& victim = g_lruList.back();
        g_weightCache.erase(victim);
        g_lruList.pop_back();
    }
}


// Compute weights for a wavelength grid and resolution parameters
std::vector<WeightSegment> compute_weights(const Vector& lam,
                                          double resOffset,
                                          double resSlope)
{
    const std::size_t n = lam.size();
    std::vector<WeightSegment> weights(n);
    
    // Precompute bin widths
    Vector dLam(n);
    dLam[0]     = lam[1]     - lam[0];
    dLam[n - 1] = lam[n - 1] - lam[n - 2];
    for (std::size_t j = 1; j < n - 1; ++j)
        dLam[j] = 0.5 * (lam[j + 1] - lam[j - 1]);
    
    const double* lamData  = lam.data();
    const double* dLamData = dLam.data();
    
    // Compute weights for each output point
    #pragma omp parallel for schedule(dynamic, 32) if (_OPENMP)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i)
    {
        const double λ_i   = lamData[i];
        const double R     = resOffset + resSlope * λ_i;
        const double sigma = (λ_i / R) * SIGMA_FROM_FWHM;
        
        const double lamMin = λ_i - KERNEL_RADIUS * sigma;
        const double lamMax = λ_i + KERNEL_RADIUS * sigma;
        
        const std::size_t jStart = std::lower_bound(lamData, lamData + n, lamMin) - lamData;
        const std::size_t jEnd   = std::upper_bound(lamData + jStart, lamData + n, lamMax) - lamData;
        
        weights[i].jStart = jStart;
        weights[i].jEnd = jEnd;
        weights[i].weights.resize(jEnd - jStart);
        
        const double inv2σ2 = 1.0 / (2.0 * sigma * sigma);
        double wSum = 0.0;
        
        // Compute normalized weights
        for (std::size_t j = jStart; j < jEnd; ++j) {
            double delta = lamData[j] - λ_i;
            double w = std::exp(-delta * delta * inv2σ2) * dLamData[j];
            weights[i].weights[j - jStart] = w;
            wSum += w;
        }
        
        // Normalize weights
        const double invWSum = 1.0 / wSum;
        for (auto& w : weights[i].weights) {
            w *= invWSum;
        }
    }
    
    return weights;
}

// Create cache key from wavelength grid and resolution parameters
CacheKey make_cache_key(const Vector& lam, double resOffset, double resSlope) {
    CacheKey key;
    key.lamStart = lam[0];
    key.lamEnd = lam[lam.size() - 1];
    key.nPoints = lam.size();
    key.avgDelta = (key.lamEnd - key.lamStart) / (key.nPoints - 1);
    key.resOffset = resOffset;
    key.resSlope = resSlope;
    return key;
}

} // anonymous namespace

// Fast application of precomputed weights
static Vector apply_weights(const Vector& flux, const std::vector<WeightSegment>& weights) {
    const std::size_t n = weights.size();
    Vector out(n);
    const double* fluxData = flux.data();
    
    #pragma omp parallel for schedule(static) if (_OPENMP)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
        const auto& seg = weights[i];
        double sum = 0.0;
        
        // Simple dot product with precomputed normalized weights
        #pragma omp simd reduction(+:sum)
        for (std::size_t j = 0; j < seg.weights.size(); ++j) {
            sum += seg.weights[j] * fluxData[seg.jStart + j];
        }
        
        out[i] = sum;
    }
    
    return out;
}

// Main resolution degradation function with caching
Vector degrade_resolution(const Vector& lam,
                         const Vector& flux,
                         double resOffset,
                         double resSlope)
{
#ifdef DIGGA_USE_CUDA
    return degrade_resolution_cuda(lam, flux, resOffset, resSlope);
#else
    // Create cache key
    CacheKey key = make_cache_key(lam, resOffset, resSlope);
    
    // Check if weights are cached
    {
        std::lock_guard<std::mutex> lock(g_cacheMutex);
        auto it = g_weightCache.find(key);
        if (it != g_weightCache.end()) {
            touch_key(key);
            return apply_weights(flux, it->second);
        }
    }
    
    // Compute weights
    auto weights = compute_weights(lam, resOffset, resSlope);

    // Insert into cache with LRU enforcement
    {
        std::lock_guard<std::mutex> lock(g_cacheMutex);
        insert_cache_entry(key, std::move(weights));
    }

    // Apply weights
    return apply_weights(flux, g_weightCache[key]);

#endif
}

// Optional: Function to precompute weights for known grids
void precompute_weights(const Vector& lam, double resOffset, double resSlope) {
    CacheKey key = make_cache_key(lam, resOffset, resSlope);
    
    std::lock_guard<std::mutex> lock(g_cacheMutex);
    if (g_weightCache.find(key) == g_weightCache.end()) {
        g_weightCache[key] = compute_weights(lam, resOffset, resSlope);
    }
}

// Optional: Clear cache if memory becomes an issue
void clear_weight_cache() {
    std::lock_guard<std::mutex> lock(g_cacheMutex);
    g_weightCache.clear();
}

// Optional: Get cache statistics
std::size_t get_cache_size() {
    std::lock_guard<std::mutex> lock(g_cacheMutex);
    return g_weightCache.size();
}

} // namespace specfit