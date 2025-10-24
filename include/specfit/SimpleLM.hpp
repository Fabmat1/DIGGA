#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>

namespace specfit {

/* ---------------------------  user visible bits  --------------------------- */
/*  A value ≤ 0 means "determine automatically".                              */

struct LMSolverOptions {
    int    max_iterations        = 200;      // hard upper limit
    double gradient_tolerance    = 0;        // auto
    double step_tolerance        = 0;        // auto
    double chi2_tolerance        = 0;        // auto
    double initial_lambda        = 0;        // auto
    bool   verbose               = false;    // chatty?
};

struct LMSolverSummary {
    int    iterations         = 0;
    double initial_chi2       = 0.0;
    double final_chi2         = 0.0;
    bool   converged          = false;
    std::vector<double> param_uncertainties;   // 1-σ; 0 = fixed
};

/* --------------------  internal helper (column selection)  ----------------- */

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

struct LMWorkspace
{
    Eigen::MatrixXd Jf;
    Eigen::MatrixXd JTJ;
    Eigen::VectorXd diag_JTJ;
    Eigen::VectorXd g;
    Eigen::VectorXd dx_free;
    Eigen::VectorXd dx;

    void resize(std::ptrdiff_t m,
                std::ptrdiff_t n_free,
                std::ptrdiff_t n_full)
    {
        /* grow when necessary … */
        if (Jf.rows()  < m      || Jf.cols()  < n_free)
            Jf.resize  (std::max<std::ptrdiff_t>(Jf.rows(),  m),
                        std::max<std::ptrdiff_t>(Jf.cols(), n_free));

        if (JTJ.rows() < n_free || JTJ.cols() < n_free)
            JTJ.resize (n_free, n_free);            // square, so grow once

        if (diag_JTJ.size() < n_free) diag_JTJ.resize(n_free);
        if (g.size()        < n_free) g.resize       (n_free);
        if (dx_free.size()  < n_free) dx_free.resize (n_free);
        if (dx.size()       < n_full) dx.resize      (n_full);

        /* … and then shrink logically to the exact size that is
           required for the *current* problem instance (no re-alloc): */
        Jf.conservativeResize (m, n_free);
        JTJ.conservativeResize(n_free, n_free);
        diag_JTJ.conservativeResize(n_free);
        g.conservativeResize       (n_free);
        dx_free.conservativeResize (n_free);
        dx.conservativeResize      (n_full);
    }
};

/* -------------------  Levenberg–Marquardt driver routine  ------------------ */

template<typename Functor>
LMSolverSummary
levenberg_marquardt(Functor&&                    func,
                    Eigen::VectorXd&             x,
                    const std::vector<bool>&     free_mask,
                    const std::vector<double>&   lower,
                    const std::vector<double>&   upper,
                    const LMSolverOptions&       user_opt = {},
                    LMWorkspace*                 work      = nullptr)
{
    LMSolverSummary summ;
    const int n = static_cast<int>(x.size());
    LMSolverOptions opt = user_opt;               // mutable copy

    if (opt.verbose){
        std::cout << "[LM] Entered LevMarq Function" << std::endl;
        std::cout << "[LM] Mapping parameter vector" << std::endl;
    }
        

    /* --------------------------------------------------------------- */
    /*  map full parameter vector  ->  free (variable) parameters      */
    /* --------------------------------------------------------------- */
    Eigen::VectorXi col_index;
    int n_free = 0;
    build_free_index(free_mask, col_index, n_free);
    if (n_free == 0) {                            // nothing to fit
        std::cout << "[LM]  Warning: All parameters are frozen! There is nothing to fit..." << std::endl;
        summ.converged  = true;
        summ.final_chi2 = 0.0;
        return summ;
    }

    if (opt.verbose)
        std::cout << "[LM] Mapped Parameter Vector. First model evaluation" << std::endl;
    
    /* --------------------------------------------------------------- */
    /*  first model evaluation                                         */
    /* --------------------------------------------------------------- */
    Eigen::VectorXd r;
    Eigen::MatrixXd J;
    if (opt.verbose)
            std::cout << "[LM] Function Evaluation" << std::endl;
    func(x, &r, &J);
    if (opt.verbose)
            std::cout << "[LM] Evaluation done." << std::endl;

    const std::size_t m = static_cast<std::size_t>(r.size());
    double chi2 = r.squaredNorm();
    summ.initial_chi2 = chi2;

    if (opt.verbose)
        std::cout << "[LM] Evaluated Model once. Determining initial λ and tolerances." << std::endl;

    /* --------------------------------------------------------------- */
    /*  automatic tolerances and initial λ                             */
    /* --------------------------------------------------------------- */
    Eigen::VectorXd g0 = J.transpose() * r;
    double gmax0       = g0.cwiseAbs().maxCoeff();
    const double eps   = std::numeric_limits<double>::epsilon();

    if (opt.gradient_tolerance <= 0.0) {
        /* derive it from the actual gradient, guard against gmax0 == 0 */
        if (gmax0 > 0.0)
            opt.gradient_tolerance = 1e-4 * gmax0;          // 1e-4 × |∇χ²|ₘₐₓ
        else
            opt.gradient_tolerance = 1e-12;                 // tiny fallback
    }

    double xnorm = x.lpNorm<Eigen::Infinity>();
    if (opt.step_tolerance <= 0.0)
        opt.step_tolerance = 1e-8 * std::max(1.0, xnorm);

    if (opt.chi2_tolerance  <= 0.0)
        opt.chi2_tolerance  = 1e-8 * std::max(1.0, chi2);

    if (opt.initial_lambda  <= 0.0) {
        Eigen::VectorXd diag = (J.transpose() * J).diagonal();
        opt.initial_lambda   = 1e-3 * diag.maxCoeff();
        if (opt.initial_lambda == 0.0) opt.initial_lambda = 1e-3;
    }
    double lambda = opt.initial_lambda;

    if (opt.verbose)
        std::cout << "[LM] Determinined initial λ and tolerances. One time allocations." << std::endl;


    /* --------------------------------------------------------------- */
    /*  one-time allocations                                           */
    /* --------------------------------------------------------------- */
    Eigen::MatrixXd Jf_local, JTJ_local;
    Eigen::VectorXd diag_local, g_local, dx_free_local, dx_local;

    /* pick the storage that will actually be used */
    Eigen::MatrixXd &Jf        = work ? work->Jf        : Jf_local;
    Eigen::MatrixXd &JTJ       = work ? work->JTJ       : JTJ_local;
    Eigen::VectorXd &diag_JTJ  = work ? work->diag_JTJ  : diag_local;
    Eigen::VectorXd &g         = work ? work->g         : g_local;
    Eigen::VectorXd &dx_free   = work ? work->dx_free   : dx_free_local;
    Eigen::VectorXd &dx        = work ? work->dx        : dx_local;

    /* make sure the arrays are big enough (may allocate once) */
    if (work) work->resize(m, n_free, n);
    else {                 // original behaviour
        Jf.resize (m, n_free);
        JTJ.resize(n_free, n_free);
        diag_JTJ.resize(n_free);
        g.resize(n_free);
        dx_free.resize(n_free);
        dx.resize(n);
    }

    if (opt.verbose)
        std::cout << "[LM] One time allocations complete." << std::endl;
    std::cout << "[LM]  Entering iteration loop..." << std::endl;

    /* --------------------------------------------------------------- */
    /*  main iteration loop                                            */
    /* --------------------------------------------------------------- */
    for (int it = 0; it < opt.max_iterations; ++it) {
        summ.iterations = it + 1;

        if (opt.verbose)
            std::cout << "[LM] Building Reduced Jacobian." << std::endl;

        /* ----- build reduced Jacobian (copy only the free columns) -- */
        for (int j = 0; j < n; ++j) {
            int col = col_index[j];
            if (col >= 0) Jf.col(col).noalias() = J.col(j);
        }

        if (opt.verbose)
            std::cout << "[LM] Built Reduced Jacobian. Transposing." << std::endl;

        /* ---------------------- g = Jᵀ r --------------------------- */
        g.noalias() = Jf.transpose() * r;
        double gmax = g.cwiseAbs().maxCoeff();
        if (gmax < opt.gradient_tolerance) {       // gradient small
            if (summ.iterations == 1) {            // happens immediately
                std::cout << "[LM]  Warning: gradient tolerance may be too large – "
                             "solver stopped without iterating\n";
            }
            summ.converged = true;
            break;
        }

        if (opt.verbose)
            std::cout << "[LM] Transposed. Multiplying" << std::endl;

        /* ---------- JTJ = JᵀJ  (use rank-update, lower triangle) ---- */
        JTJ.setZero();
        JTJ.selfadjointView<Eigen::Lower>().rankUpdate(Jf.adjoint(), 1.0);
        JTJ.template triangularView<Eigen::StrictlyUpper>() =
            JTJ.transpose();                       // copy to upper

        diag_JTJ = JTJ.diagonal();

        if (opt.verbose)
            std::cout << "[LM] Multiplied Matrices. Diagonalizing and solving." << std::endl;

        /* ------- (JTJ + λ D) Δx = −g   (D = diag(JTJ)) -------------- */
        JTJ.diagonal().array() +=
            lambda * (diag_JTJ.array() + 1e-20);   // Fletcher scaling

        dx_free = -JTJ.ldlt().solve(g);            // SPD solve

        if (dx_free.hasNaN() || !dx_free.allFinite()){   // numerical failure
            std::cout << "[LM]  Warning: numerical failure! Inf/NaN in solver. Aborting iteration...\n";
            break;
        }

        if (opt.verbose)
            std::cout << "[LM] Diagonalized. Copying step." << std::endl;

        /* --------------- copy step into full parameter vector ------- */
        dx.setZero();
        for (int j = 0; j < n; ++j) {
            int col = col_index[j];
            if (col >= 0) dx[j] = dx_free[col];
        }

        if (dx.cwiseAbs().maxCoeff() < opt.step_tolerance) {
            if (summ.iterations == 1) {
                std::cout << "[LM]  Warning: step tolerance too large – "
                             "tightening it once.\n";
                opt.step_tolerance *= 0.01;    // tighten and try again
            }
            else {
                summ.converged = true;         // later: genuine convergence
                break;
            }
        }

        if (opt.verbose)
            std::cout << "[LM] Copied step. Calculating candidate." << std::endl;

        /* --------------------- candidate point ---------------------- */
        Eigen::VectorXd x_try = x + dx;

        /* ---------- simple bound constraints (project) -------------- */
        for (int j = 0; j < n; ++j) {
            if (!lower.empty()) x_try[j] = std::max(x_try[j], lower[j]);
            if (!upper.empty()) x_try[j] = std::min(x_try[j], upper[j]);
        }

        /* -------- recompute dx_free based on actual step taken ------ */
        Eigen::VectorXd dx_actual = x_try - x;
        for (int j = 0; j < n; ++j) {
            int col = col_index[j];
            if (col >= 0) dx_free[col] = dx_actual[j];
        }

        Eigen::VectorXd r_try;
        Eigen::MatrixXd J_try;
        if (opt.verbose)
            std::cout << "[LM] Function Evaluation" << std::endl;
        func(x_try, &r_try, &J_try);
        if (opt.verbose)
            std::cout << "[LM] Evaluation done." << std::endl;
        double chi2_try = r_try.squaredNorm();

        if (opt.verbose)
            std::cout << "[LM] Calculated candidate. Powell test." << std::endl;

        /* ------------------- Powell's ρ test ------------------------ */
        Eigen::VectorXd tmp = lambda * (diag_JTJ.array() * dx_free.array()).matrix() - g;
        double pred_red = 0.5 * dx_free.dot(tmp);
        if (pred_red <= 0.0) pred_red = eps;

        double rho     = (chi2 - chi2_try) / pred_red;
        bool   accept  = rho > 0.0 && chi2_try < chi2;

        if (accept) {
            /* --------------- successful iteration ------------------ */
            x.swap(x_try);
            r.swap(r_try);
            J.swap(J_try);
            chi2 = chi2_try;

            /* adaptive λ (MINPACK style) ---------------------------- */
            double fac = std::max(1.0/3.0,
                                  1.0 - std::pow(2.0*rho - 1.0, 3.0));
            lambda *= fac;
            lambda  = std::max(lambda, 1e-18);

            std::cout << "[LM]  iter " << it
                        << "  ρ="  << std::fixed << std::setprecision(2) << rho
                        << "  χ²=" << std::fixed << std::setprecision(2) << chi2
                        << "  λ="  << std::fixed << std::setprecision(2) << lambda
                        << "  (accepted)\n";

            if (std::abs(pred_red) < opt.chi2_tolerance) {
                summ.converged = true;
                break;
            }
        } else {
            /* ------------------- rejected step --------------------- */
            lambda *= 2.0;
            std::cout << "[LM]  iter " << it
                        << "  ρ="  << std::fixed << std::setprecision(2) << rho
                        << "  χ²=" << std::fixed << std::setprecision(2) << chi2_try
                        << "  λ="  << std::fixed << std::setprecision(2) << lambda
                        << "  (rejected)\n";
        }
    }

    summ.final_chi2 = chi2;

    if (opt.verbose)
            std::cout << "[LM] Iter complete, getting uncertainties." << std::endl;

    /* ---------------------  propagate uncertainties  ---------------- */
    summ.param_uncertainties.assign(n, 0.0);
    if (n_free > 0) {
        /* reuse Jf and JTJ already allocated ------------------------- */
        for (int j = 0; j < n; ++j) {
            int col = col_index[j];
            if (col >= 0) Jf.col(col).noalias() = J.col(j);
        }
        JTJ.setZero();
        JTJ.selfadjointView<Eigen::Lower>().rankUpdate(Jf.adjoint(), 1.0);
        JTJ.template triangularView<Eigen::StrictlyUpper>() =
            JTJ.transpose();

        const double dof = std::max<std::size_t>(m - n_free, 1);
        const double var = r.squaredNorm() / dof;             // σ² ≈ χ²/dof

        Eigen::MatrixXd cov =
            JTJ.ldlt().solve(Eigen::MatrixXd::Identity(n_free, n_free));
        cov *= var;

        for (int j = 0; j < n; ++j) {
            int col = col_index[j];
            if (col >= 0)
                summ.param_uncertainties[j] =
                    std::sqrt(std::max(0.0, cov(col, col)));
        }
    }

    if (opt.verbose)
            std::cout << "[LM] LevMarq done." << std::endl;
    return summ;
}

} // namespace specfit