#include "specfit/Rebin.hpp"
#include <algorithm>
#include <numeric>

namespace specfit {

/* -------------------------------------------------------------- *
 *  helper: linear interpolation of y(x) for a *monotonic* array  *
 * -------------------------------------------------------------- */
static double interp(const Vector& x, const Vector& y, double xi)
{
    if (xi <= x[0])               return y[0];
    if (xi >= x[x.size() - 1])    return y[y.size() - 1];

    const auto it = std::lower_bound(x.data(), x.data() + x.size(), xi);
    const int  hi = static_cast<int>(it - x.data());
    const int  lo = hi - 1;
    const double w = (xi - x[lo]) / (x[hi] - x[lo]);
    return y[lo] * (1.0 - w) + y[hi] * w;
}

/* -------------------------------------------------------------- *
 *  build pixel edges from centres:                               *
 *      e_0 , c_0 , e_1 , c_1 , …                                 *
 * -------------------------------------------------------------- */
static Vector make_edges(const Vector& centres)
{
    const int N = centres.size();
    Vector edges(N + 1);

    edges[0]     = centres[0] - 0.5 * (centres[1] - centres[0]);
    for (int i = 1; i < N; ++i)
        edges[i] = 0.5 * (centres[i - 1] + centres[i]);
    edges[N]     = centres[N - 1] + 0.5 * (centres[N - 1] - centres[N - 2]);

    return edges;
}

/* -------------------------------------------------------------- *
 *  integrate f(λ) once → F(λ)  and interpolate that integral      *
 * -------------------------------------------------------------- */
static Vector cumulative_trapz(const Vector& x, const Vector& y)
{
    const int N = x.size();
    Vector F(N);
    F[0] = 0.0;
    for (int i = 1; i < N; ++i)
        F[i] = F[i - 1] +
               0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]);
    return F;
}

/* ==============================================================
 *  public interface                                             
 * =============================================================*/
Vector trapezoidal_rebin(const Vector& lam_in,
                         const Vector& flux_in,
                         const Vector& lam_out)
{
    /* ---- derive pixel edges ---- */
    const Vector in_edges  = make_edges(lam_in);
    const Vector out_edges = make_edges(lam_out);

    /* ---- cumulative integral of input spectrum ---- */
    const Vector F = cumulative_trapz(lam_in, flux_in);

    /* helper that returns ∫ f dλ  from λ_0  to  xi               */
    auto integral_at = [&](double xi) -> double {
        if (xi <= lam_in[0])            return 0.0;
        if (xi >= lam_in[lam_in.size()-1])
            return F[F.size() - 1];

        const auto it = std::lower_bound(lam_in.data(),
                                         lam_in.data() + lam_in.size(), xi);
        const int hi = static_cast<int>(it - lam_in.data());
        const int lo = hi - 1;

        /* partial area from lam_in[lo]  to  xi                    */
        const double f_lo = flux_in[lo];
        const double f_hi = interp(lam_in, flux_in, xi);
        const double dx   = xi - lam_in[lo];
        const double area = 0.5 * (f_lo + f_hi) * dx;

        return F[lo] + area;
    };

    /* ---- rebin by evaluating the integral at the edges ---- */
    Vector out(lam_out.size());
    for (int i = 0; i < lam_out.size(); ++i) {
        const double lo = out_edges[i];
        const double hi = out_edges[i + 1];

        const double area = integral_at(hi) - integral_at(lo);
        double w = hi - lo;
        out[i] = (w > 0.0) ? area / w
                           : flux_in[std::min<std::ptrdiff_t>(
                                       lam_in.size() - 1,
                                       std::lower_bound(lam_in.data(),
                                                        lam_in.data()+lam_in.size(), lo)
                                     - lam_in.data())];
    }
    return out;
}

} // namespace specfit