# Continuous Regularized Wasserstein Barycenters
Lingxiao Li, [Aude Genevay](https://audeg.github.io/), [Mikhail Yurochkin](https://moonfolk.github.io/), [Justin Solomon](https://people.csail.mit.edu/jsolomon/)
[[arXiv](https://arxiv.org/abs/2008.12534)]

NeurIPS 2020

## Citation
```
@misc{2008.12534,
Author = {Lingxiao Li and Aude Genevay and Mikhail Yurochkin and Justin Solomon},
Title = {Continuous Regularized Wasserstein Barycenters},
Year = {2020},
Eprint = {arXiv:2008.12534},
}
```

## Abstract
Wasserstein barycenters provide a geometrically meaningful way to aggregate probability distributions, built on the theory of optimal transport. They are difficult to compute in practice, however, leading previous work to restrict their supports to finite sets of points. Leveraging a new dual formulation for the regularized Wasserstein barycenter problem, we introduce a stochastic algorithm that constructs a continuous approximation of the barycenter. We establish strong duality and use
the corresponding primal-dual relationship to parameterize the barycenter implicitly using the dual potentials of regularized transport problems. The resulting problem can be solved with stochastic gradient descent, which yields an efficient online algorithm to approximate the Wasserstein barycenter of continuous distributions given sample access. We demonstrate the effectiveness of our approach and compare against previous work on both synthetic examples and real-world
applications.

## Dependencies and Installation
The following python packages are required:
- tensorflow (>= 2.1.0)
- pyyaml
- importlib_resources (if python version &lt; 3.7)
- tensorflow_probability
- pandas
- sklearn
- POT (for comparison)
- matplotlib (for visualization)

To run the code, you first need to install the package locally, via `pip install -e /path/to/this_package`.

## Code structure
* `cwb/` contains the core code for the stochastic barycenter algorithm and the experiments setup scripts.
* `experiments/` contains configurations for the qualitative experiments in the paper.
* `bike_trips_sampler/` contains the scripts to generate posterior samples for the subset posterior aggregation experiment.

### YAML config file
The configuration of an experiment is entirely described by a YAML config file.
See `experiments/qualitative/` for examples.
See `cwb/common/config_parser.py` to find out exactly how the config file is parsed and what the options are.

### Running the qualitative experiments
To run each of the qualitative experiments, use the following commands (taking the annulus and square example):
```
cd experiments/qualitative/annulus_square/
python -m cwb.barycenter --train --test config.yaml
```
The option `--train` will train the dual potentials for the barycenter. It will generate various validation files at certain steps, depending on the `val_entries` in the config file.

The option `--test` will perform each test time jobs (such as sampling from the barycenter of run MCMC to get the barycenter marginal samples) specified under `test` in the config file.

All paths in the config file are relative to the current working directory (by default it is where you run `python ...`).

To visualize the 2D results (with the additional required packages), simply run
```
python -m cwb.barycenter --visualize config.yaml
```
This will generate a bunch of new folders containing visualization results in various format (pictures or videos, depending on the configuration).

### Running the Gaussian experiments with varying dimensions
First create a folder for the experiment:
```
mkdir experiments/gaussian
cd experiments/gaussian
```
Next generate data (`--dims` indicates which dimensions to generate data for):
```
python -m cwb.tests.comparison.batch gaussian --gen_data --dims 2 3 4 5 6 7 8
```
Then calculate ground truth barycenter using a [fixed-point algorithm](https://arxiv.org/abs/1511.05355):
```
python -m cwb.tests.comparison.batch gaussian --run gaussian_iterative --dims 2 3 4 5 6 7 8
```
To run our barycenter algorithm, use (set `--repeat_times` to repeat the experiments for multiple times, and `--reseed` to refresh random seeds based on time)
```
python -m cwb.tests.comparison.batch gaussian --run cwb --dims 2 3 4 5 6 7 8 --repeat_start=0 --repeat_times=5 --reseed
```
To calculate the statistics, use
```
python -m cwb.tests.comparison.batch gaussian --validate cwb --dims 2 3 4 5 6 7 8 --repeat_start=0 --repeat_times=5 --reseed
```
Finally to display the statistics as a LaTeX table, run
```
python -m cwb.tests.comparison.latexify gaussian cwb --dims 2 3 4 5 6 7 8 --repeat_start=0 --repeat_times=5 --losses fit_gaussian_mean_loss fit_gaussian_cov_loss W2_lp
```
See `cwb/tests/comparison/validate.py` and `cwb/tests/comparison/latexify.py` for how to include different evaluation metrics.

### Running the subset posterior aggregation experiment
First create a folder for the experiment:
```
mkdir experiments/poisson
cd experiments/poisson
```
Next generate data:
```
python -m cwb.tests.comparison.batch poisson --gen_data --dims 8
```
Running and testing commands will be similar to those for the Gaussian experiments.
To run our barycenter algorithm, use
```
python -m cwb.tests.comparison.batch poisson --run cwb --dims 8 --repeat_start=0 --repeat_times=20 --reseed
```
To calculate the statistics, use
```
python -m cwb.tests.comparison.batch poisson --validate cwb --dims 8 --repeat_start=0 --repeat_times=20 --reseed
```
Finally to display the statistics as a LaTeX table, run
```
python -m cwb.tests.comparison.latexify poisson cwb --dims 8 --repeat_start=0 --repeat_times=20 --losses mm_mean_loss  mm_cov_loss W2_lp
```

### License
This code is released under the MIT License. Refer to LICENSE for details.
