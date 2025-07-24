 #pragma once
#include "Types.hpp"
#include "CommonTypes.hpp"
#include "UnifiedFitWorkflow.hpp"
#include <string>
#include <vector>

namespace specfit {

/* --------------------------------------------------------------------- */
/*                     L a T e X  parameter table                        */
/* --------------------------------------------------------------------- */
class PdfReport {
public:
    explicit PdfReport(const std::string& path);
    void add_parameter(const std::string& name,
                       double              value,
                       double              error,
                       const std::string&  scope);
    void write_and_compile();
private:
    std::string              path_;
    std::vector<std::string> lines_;
};

/* --------------------------------------------------------------------- */
/*                Multi-panel diagnostic spectrum plots                  */
/* --------------------------------------------------------------------- */
class MultiPanelPlotter
{
  public:
      explicit MultiPanelPlotter(double xrange, bool grey=false);
      /* plot the whole spectrum ------------------------------------------------*/
      void plot(const std::string& pdf_path,
                const Spectrum&    spec,        /* NEW */
                const Vector&      model,       /* continuum–free model          */
                const Vector&      continuum)   /* continuum spline              */
                const;

      /* a quick two-line overlay ----------------------------------------------*/
      void simple_plot(const std::string& pdf,
                       const Spectrum&    spec,  /* NEW */
                       const Vector&      model) const;
    private:
        double xrange_;
        bool   grey_;
};

/* --------------------------------------------------------------------- */
/*      High-level helper:  create *all* results at the very end         */
/* --------------------------------------------------------------------- */
void generate_results(const std::string&   out_dir,
                      const UnifiedFitWorkflow& wf,
                      const std::vector<DataSet>& datasets,
                      const SharedModel&   model,
                      double               xrange,
                      bool                 grey,
                      const std::vector<std::string>& untied_params);

} // namespace specfit