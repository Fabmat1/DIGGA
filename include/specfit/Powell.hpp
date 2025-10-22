#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>

namespace specfit {

/* ---------------------------  user visible bits  --------------------------- */

struct PowellSolverOptions {
    int    max_iterations        = 200;      // hard upper limit
    int    max_function_evals    = 50000;      // max function evaluations
    double relative_tolerance    = 1e-4;     // relative function tolerance
    double absolute_tolerance    = 1e-9;     // absolute function tolerance
    double step_tolerance        = 0;        // auto
    bool   verbose              = false;     // chatty?
};

struct PowellSolverSummary {
    int    iterations         = 0;
    int    function_evals     = 0;
    double initial_value      = 0.0;
    double final_value        = 0.0;
    bool   converged          = false;
};

/* ------------------------  internal helper functions  ----------------------- */

namespace detail {

struct LineSearchData {
    Eigen::VectorXd p;          // current point
    Eigen::VectorXd xi;         // search direction
    const Eigen::VectorXd& min_p;
    const Eigen::VectorXd& max_p;
    std::function<double(const Eigen::VectorXd&)> func;
};

inline double call_func_1d(double lambda, const LineSearchData& data)
{
    Eigen::VectorXd p_trial = data.p + lambda * data.xi;
    
    // Enforce bounds
    for (int i = 0; i < p_trial.size(); ++i) {
        if (p_trial[i] < data.min_p[i] || p_trial[i] > data.max_p[i])
            return std::numeric_limits<double>::infinity();
    }
    
    return data.func(p_trial);
}

inline std::tuple<double, double> compute_lambda_bounds(
    const Eigen::VectorXd& p,
    const Eigen::VectorXd& xi,
    const Eigen::VectorXd& min_p,
    const Eigen::VectorXd& max_p)
{
    const int n = p.size();
    double min_lambda = -std::numeric_limits<double>::infinity();
    double max_lambda = std::numeric_limits<double>::infinity();
    
    for (int i = 0; i < n; ++i) {
        if (std::abs(xi[i]) > std::numeric_limits<double>::epsilon()) {
            double t1 = (min_p[i] - p[i]) / xi[i];
            double t2 = (max_p[i] - p[i]) / xi[i];
            
            if (xi[i] > 0) {
                min_lambda = std::max(min_lambda, t1);
                max_lambda = std::min(max_lambda, t2);
            } else {
                min_lambda = std::max(min_lambda, t2);
                max_lambda = std::min(max_lambda, t1);
            }
        }
    }
    
    return {min_lambda, max_lambda};
}

inline std::tuple<double, double> line_minimize(
    const LineSearchData& data,
    double initial_step,
    double f0,
    int& nfe,
    int max_nfe,
    bool verbose)
{
    const double eps = std::numeric_limits<double>::epsilon();
    const double abserr = std::sqrt(eps) * data.p.norm();
    
    auto [min_lambda, max_lambda] = compute_lambda_bounds(
        data.p, data.xi, data.min_p, data.max_p);
    
    if (min_lambda >= max_lambda) {
        return {0.0, f0};
    }
    
    // Bracket the minimum
    double x0 = 0.0;
    double x1 = initial_step;
    
    // Ensure x1 is within bounds
    x1 = std::max(min_lambda, std::min(max_lambda, x1));
    
    double f1 = call_func_1d(x1, data);
    nfe++;
    
    if (nfe >= max_nfe || std::isinf(f1)) {
        return {0.0, f0};
    }
    
    double xmin = x0, fmin = f0;
    if (f1 < fmin) {
        xmin = x1;
        fmin = f1;
    }
    
    // Try to bracket the minimum
    double x2;
    if (f1 < f0) {
        x2 = x1 + (x1 - x0);
    } else {
        x2 = x0 - (x1 - x0);
    }
    
    x2 = std::max(min_lambda, std::min(max_lambda, x2));
    
    if (std::abs(x2 - x1) < abserr || std::abs(x2 - x0) < abserr) {
        return {xmin, fmin};
    }
    
    double f2 = call_func_1d(x2, data);
    nfe++;
    
    if (nfe >= max_nfe || std::isinf(f2)) {
        return {xmin, fmin};
    }
    
    if (f2 < fmin) {
        xmin = x2;
        fmin = f2;
    }
    
    // Quadratic interpolation
    int iter = 0;
    const int max_iter = 20;
    
    while (iter++ < max_iter && nfe < max_nfe) {
        // Sort points by x-coordinate
        if (x1 > x2) {
            std::swap(x1, x2);
            std::swap(f1, f2);
        }
        if (x0 > x1) {
            std::swap(x0, x1);
            std::swap(f0, f1);
        }
        if (x1 > x2) {
            std::swap(x1, x2);
            std::swap(f1, f2);
        }
        
        // Check if we have a bracket
        if (f0 > f1 && f1 < f2) {
            // Quadratic interpolation
            double denom = 2.0 * ((f2-f1)/(x2-x1) - (f1-f0)/(x1-x0));
            if (std::abs(denom) > eps) {
                double xnew = 0.5*(x0+x1) - ((f1-f0)/(x1-x0))/denom;
                xnew = std::max(min_lambda, std::min(max_lambda, xnew));
                
                if (std::abs(xnew - x1) > abserr && 
                    std::abs(xnew - x0) > abserr && 
                    std::abs(xnew - x2) > abserr) {
                    
                    double fnew = call_func_1d(xnew, data);
                    nfe++;
                    
                    if (fnew < fmin) {
                        xmin = xnew;
                        fmin = fnew;
                    }
                    
                    // Update bracket
                    if (xnew < x1) {
                        if (fnew < f1) {
                            x2 = x1; f2 = f1;
                            x1 = xnew; f1 = fnew;
                        } else {
                            x0 = xnew; f0 = fnew;
                        }
                    } else {
                        if (fnew < f1) {
                            x0 = x1; f0 = f1;
                            x1 = xnew; f1 = fnew;
                        } else {
                            x2 = xnew; f2 = fnew;
                        }
                    }
                    continue;
                }
            }
        }
        
        // Golden section search fallback
        double golden = 0.38197;  // (3 - sqrt(5))/2
        double xnew;
        
        if (f0 > f2) {
            xnew = x1 + golden * (x2 - x1);
            x0 = x1; f0 = f1;
            x1 = x2; f1 = f2;
        } else {
            xnew = x1 - golden * (x1 - x0);
            x2 = x1; f2 = f1;
            x1 = x0; f1 = f0;
        }
        
        xnew = std::max(min_lambda, std::min(max_lambda, xnew));
        
        if (std::abs(xnew - xmin) < abserr) {
            break;
        }
        
        double fnew = call_func_1d(xnew, data);
        nfe++;
        
        if (fnew < fmin) {
            xmin = xnew;
            fmin = fnew;
        }
        
        if (f0 > f2) {
            x2 = xnew; f2 = fnew;
        } else {
            x0 = xnew; f0 = fnew;
        }
        
        if (std::abs(x2 - x0) < abserr) {
            break;
        }
    }
    
    return {xmin, fmin};
}

inline bool converged(double f_old, double f_new, 
                     double reltol, double abstol)
{
    double diff = std::abs(f_new - f_old);
    double scale = std::max(std::abs(f_old), std::abs(f_new));
    return (diff <= abstol) || (diff <= reltol * scale);
}

} // namespace detail

/* -------------------  Powell's method driver routine  ---------------------- */

template<typename Functor>
PowellSolverSummary
powell(Functor&&                    func,
       Eigen::VectorXd&             x,
       const std::vector<bool>&     free_mask,
       const std::vector<double>&   lower,
       const std::vector<double>&   upper,
       const PowellSolverOptions&   user_opt = {})
{
    PowellSolverSummary summ;
    const int n_full = static_cast<int>(x.size());
    
    // Build mapping for free parameters
    Eigen::VectorXi col_index;
    int n_free = 0;
    build_free_index(free_mask, col_index, n_free);
    
    if (n_free == 0) {
        if (user_opt.verbose)
            std::cout << "[Powell] Warning: All parameters are frozen!\n";
        summ.converged = true;
        return summ;
    }
    
    // Set up bounds
    Eigen::VectorXd min_p(n_full), max_p(n_full);
    for (int i = 0; i < n_full; ++i) {
        min_p[i] = lower.empty() ? -1e10 : lower[i];
        max_p[i] = upper.empty() ?  1e10 : upper[i];
    }
    
    // Ensure initial point is within bounds
    for (int i = 0; i < n_full; ++i) {
        x[i] = std::max(min_p[i], std::min(max_p[i], x[i]));
    }
    
    // Wrapper for objective function (no derivatives needed)
    auto eval_func = [&func](const Eigen::VectorXd& p) -> double {
        Eigen::VectorXd r;
        func(p, &r, nullptr);
        return r.squaredNorm();
    };
    
    // Initial function value
    double f0 = eval_func(x);
    summ.initial_value = f0;
    summ.function_evals = 1;
    
    // Initialize search directions (coordinate directions for free parameters)
    std::vector<Eigen::VectorXd> directions;
    std::vector<double> lambdas(n_free, 0.01);
    
    for (int j = 0; j < n_full; ++j) {
        if (col_index[j] >= 0) {
            Eigen::VectorXd xi = Eigen::VectorXd::Zero(n_full);
            xi[j] = 1.0;
            directions.push_back(xi);
            
            // Initial step size
            lambdas[col_index[j]] = (x[j] != 0.0) ? 0.01 * std::abs(x[j]) : 0.01;
        }
    }
    
    PowellSolverOptions opt = user_opt;
    if (opt.step_tolerance <= 0.0) {
        double xnorm = x.lpNorm<Eigen::Infinity>();
        opt.step_tolerance = 1e-8 * std::max(1.0, xnorm);
    }
    
    Eigen::VectorXd p0 = x;
    int no_progress_count = 0;
    const int max_no_progress = 2;
    
    // Main iteration loop
    for (int iter = 0; iter < opt.max_iterations; ++iter) {
        summ.iterations = iter + 1;
        
        if (summ.function_evals >= opt.max_function_evals) {
            if (opt.verbose)
                std::cout << "[Powell] Max function evaluations reached\n";
            break;
        }
        
        if (opt.verbose) {
            std::cout << "[Powell] iter " << iter 
                     << " f=" << f0 
                     << " nfe=" << summ.function_evals << "\n";
        }
        
        Eigen::VectorXd pn = p0;
        double fn = f0;
        
        // Minimize along each search direction
        for (int i = 0; i < n_free; ++i) {
            detail::LineSearchData ldata = {
                pn, directions[i], min_p, max_p, eval_func
            };
            
            auto [lambda, f] = detail::line_minimize(
                ldata, lambdas[i], fn, summ.function_evals,
                opt.max_function_evals - summ.function_evals,
                opt.verbose);
            
            if (lambda != 0.0) {
                pn += lambda * directions[i];
                lambdas[i] = std::abs(lambda);
                fn = f;
            }
        }
        
        // Check for convergence
        if (detail::converged(f0, fn, opt.relative_tolerance, 
                             opt.absolute_tolerance)) {
            if (++no_progress_count >= max_no_progress) {
                summ.converged = true;
                x = pn;
                summ.final_value = fn;
                break;
            }
        } else {
            no_progress_count = 0;
        }
        
        // Powell's direction update
        Eigen::VectorXd new_dir = pn - p0;
        double dir_norm = new_dir.norm();
        
        if (dir_norm > opt.step_tolerance) {
            new_dir /= dir_norm;
            
            // Line search along the new direction
            detail::LineSearchData ldata = {
                pn, new_dir, min_p, max_p, eval_func
            };
            
            auto [lambda, f] = detail::line_minimize(
                ldata, 0.5 * dir_norm, fn, summ.function_evals,
                opt.max_function_evals - summ.function_evals,
                opt.verbose);
            
            if (lambda != 0.0) {
                pn += lambda * new_dir;
                fn = f;
                
                // Replace the direction that contributed most
                auto max_it = std::max_element(lambdas.begin(), lambdas.end());
                int max_idx = std::distance(lambdas.begin(), max_it);
                
                directions[max_idx] = new_dir;
                lambdas[max_idx] = std::abs(lambda);
            }
        }
        
        p0 = pn;
        f0 = fn;
    }
    
    x = p0;
    summ.final_value = f0;
    
    if (summ.iterations >= opt.max_iterations && !summ.converged) {
        if (opt.verbose)
            std::cout << "[Powell] Max iterations reached without convergence\n";
    }
    
    return summ;
}

} // namespace specfit