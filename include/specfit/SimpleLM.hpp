#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <limits>
#include <cmath>

namespace specfit {

// ---------------------------  user visible bits  -----------------------------

struct LMSolverOptions {
    int    max_iterations        = 500;
    double gradient_tolerance    = 1e-6;
    double step_tolerance        = 1e-8;
    double chi2_tolerance        = 1e-12;
    double initial_lambda        = 1e-3;
    double lambda_up_factor      = 10.0;
    double lambda_down_factor    = 0.2;
    bool   verbose               = false;
};

struct LMSolverSummary {
    int    iterations         = 0;
    double initial_chi2       = 0.0;
    double final_chi2         = 0.0;
    bool   converged          = false;

    /* 1-σ uncertainties *for the full parameter vector* (0 if fixed) */
    std::vector<double> param_uncertainties;
};

// --------------------  internal helper (column selection)  -------------------

inline
void build_free_index(const std::vector<bool>& mask,
                      Eigen::VectorXi&         map_full_to_reduced,
                      int&                     n_free)
{
    const int n = static_cast<int>(mask.size());
    map_full_to_reduced.resize(n);
    n_free = 0;
    for (int j = 0; j < n; ++j) {
        if (mask.empty() || mask[j])
            map_full_to_reduced[j] = n_free++;
        else
            map_full_to_reduced[j] = -1;
    }
}

// -------------------  Levenberg–Marquardt driver routine  --------------------

template<typename Functor>
LMSolverSummary
levenberg_marquardt(Functor&&                    func,
                    Eigen::VectorXd&             x,          // in/out (full)
                    const std::vector<bool>&     free_mask,  // same size as x
                    const std::vector<double>&   lower,
                    const std::vector<double>&   upper,
                    const LMSolverOptions&       opt = {})
{
    LMSolverSummary summ;
    const int n = static_cast<int>(x.size());

    /* ------------------------------------------------------------------ */
    /*  build index that maps full parameter vector  ->  free parameters  */
    /* ------------------------------------------------------------------ */
    Eigen::VectorXi col_index;
    int n_free = 0;
    build_free_index(free_mask, col_index, n_free);
    if (n_free == 0) {
        summ.converged   = true;
        summ.final_chi2  = 0.0;
        return summ;
    }

    /* ------------------------------------------------------------------ */
    /*  allocate work arrays once                                         */
    /* ------------------------------------------------------------------ */
    Eigen::VectorXd r;
    Eigen::MatrixXd J;

    func(x, &r, &J);
    double chi2 = r.squaredNorm();
    summ.initial_chi2 = chi2;

    double lambda = opt.initial_lambda;

    /* ------------------------------------------------------------------ */
    /*  main iteration loop                                               */
    /* ------------------------------------------------------------------ */
    for (int it = 0; it < opt.max_iterations; ++it) {
        summ.iterations = it + 1;

        // build reduced ( m × n_free ) Jacobian that contains only columns
        Eigen::MatrixXd Jf(r.size(), n_free);
        for (int j = 0; j < n; ++j) {
            int col = col_index[j];
            if (col >= 0) Jf.col(col) = J.col(j);
        }

        Eigen::VectorXd g  = Jf.transpose() * r;             // gradient
        double gmax = g.cwiseAbs().maxCoeff();
        if (gmax < opt.gradient_tolerance) {                 // converged
            summ.converged = true;
            break;
        }

        Eigen::MatrixXd A = Jf.transpose() * Jf;
        A.diagonal().array() += lambda;                      // LM damping
        Eigen::VectorXd dx_free = -A.ldlt().solve(g);

        Eigen::VectorXd dx = Eigen::VectorXd::Zero(n);       // full step
        for (int j = 0; j < n; ++j) {
            int col = col_index[j];
            if (col >= 0) dx[j] = dx_free[col];
        }

        if (dx.cwiseAbs().maxCoeff() < opt.step_tolerance) {
            summ.converged = true;
            break;
        }

        // candidate point
        Eigen::VectorXd x_try = x + dx;

        // honour simple box bounds
        for (int j = 0; j < n; ++j) {
            if (!lower.empty()) x_try[j] = std::max(x_try[j], lower[j]);
            if (!upper.empty()) x_try[j] = std::min(x_try[j], upper[j]);
        }

        // evaluate new χ²
        Eigen::VectorXd r_try;
        Eigen::MatrixXd J_try;
        func(x_try, &r_try, &J_try);
        double chi2_try = r_try.squaredNorm();

        const bool accept = chi2_try < chi2;
        if (accept) {
            const double dchi2 = chi2 - chi2_try;
            x.swap(x_try);
            r.swap(r_try);
            J.swap(J_try);
            chi2     = chi2_try;
            lambda   = std::max(lambda * opt.lambda_down_factor, 1e-12);
            if (opt.verbose)
                std::cout << "[LM]  iter " << it
                          << "  χ²=" << chi2
                          << "  λ="  << lambda << "  (accepted)\n";
            if (std::abs(dchi2) < opt.chi2_tolerance) {
                summ.converged = true;
                break;
            }
        } else {
            lambda *= opt.lambda_up_factor;
            if (opt.verbose)
                std::cout << "[LM]  iter " << it
                          << "  χ²=" << chi2_try
                          << "  λ="  << lambda << "  (rejected)\n";
        }
    }

    summ.final_chi2 = chi2;

    /* -------------  finished → propagate uncertainties -------------- */
    summ.param_uncertainties.assign(n, 0.0);
    if (n_free > 0) {
        /* build final reduced Jacobian Jf once more ------------------- */
        Eigen::MatrixXd Jf(r.size(), n_free);
        for (int j = 0; j < n; ++j) {
            int col = col_index[j];
            if (col >= 0) Jf.col(col) = J.col(j);
        }
        Eigen::MatrixXd JTJ = Jf.transpose() * Jf;

        const double dof = std::max<int>(r.size() - n_free, 1);
        const double var = r.squaredNorm() / dof;          // σ² ≈ χ²/dof

        Eigen::MatrixXd cov = JTJ.ldlt().solve(
                                Eigen::MatrixXd::Identity(n_free, n_free));
        cov *= var;

        for (int j = 0; j < n; ++j) {
            int col = col_index[j];
            if (col >= 0)
                summ.param_uncertainties[j] =
                    std::sqrt(std::max(0.0, cov(col, col)));
        }
    }
    return summ;
}

} // namespace specfit