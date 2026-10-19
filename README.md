# GP-based multi-fidelity baselines

Code for the comparative baselines used in Chapter 6 of my PhD thesis
(*Trusted Predictive Modelling for Engineering Systems with Large Scale Data*,
University of Sheffield). The repository implements four Gaussian-process-based
approaches against which a new multi-fidelity method is compared.

**Status:** Research code. These are my own implementations and are not the original authors' code.

## Methods

| # | Method | Reference | Short description |
|---|--------|-----------|-------------------|
| 1 | Standard GP (baseline) | [4] | A GP trained on a single dataset, with no fidelity information. |
| 2 | Fidelity-weighted learning (FWL) | [1] | Teacher–student scheme: a GP teacher, fitted on high-quality data, estimates the quality of cheap labels so a student network can weight them in training. |
| 3 | Nonlinear information fusion (NARGP) | [2] | Nonlinear autoregressive multi-fidelity GP: each fidelity level is a GP whose input is augmented with the posterior from the level below, allowing nonlinear cross-fidelity relationships. |
| 4 | Deep multi-fidelity GP | [3] | Builds on the autoregressive scheme to handle more general cross-correlations between fidelities. |

## Usage

Run base_models_main, dehghani_2018, perdikaris_2017, raissi_george_2016 indepednently. 

## Data

Synthetic data was generated for comparison. 

## References

[1] M. Dehghani, A. Mehrjou, S. Gouws, J. Kamps, and B. Schölkopf,
"Fidelity-weighted learning," in *Proc. ICLR*, 2018, arXiv:1711.02799.

[2] P. Perdikaris, M. Raissi, A. Damianou, N. D. Lawrence, and G. E. Karniadakis,
"Nonlinear information fusion algorithms for data-efficient multi-fidelity
modelling," *Proc. R. Soc. A*, vol. 473, no. 2198, 20160751, 2017,
doi: 10.1098/rspa.2016.0751.

[3] M. Raissi and G. Karniadakis, "Deep multi-fidelity Gaussian processes," 2016,
arXiv:1604.07484.

[4] C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine
Learning*. MIT Press, 2006.
