# Paperfold - BNN

Paperfold is a companion toy example for our tutorial ["Hands-on Bayesian Neural Networks - A Tutorial for Deep Learning Users"](https://doi.org/10.1109/MCI.2022.3155327). It illustrate different inference methods for BNm, as well as some of their benefits and limitations, on a small model with 8 parameters (To make it easier to plot the actual samples from the posterior).

## Dependancies

The code depend on: 

- numpy (tested with version 1.19.2), 
- pandas (tested with version 1.0.2),
- pytorch (tested with version 1.8.1),
- pyro (tested with version 1.6.0),
- matplotlib (tested with version 3.1.1),
- seaborn (tested with version 0.10.0),

and two libraries from the base python distribution: argparse and time.

It has been tested with python 3.6.9.

## Usage

The project is split into multiple files. A first series of modules define the models:

- numpyModel contain the model implemented using numpy primitives. This allows to use the samples generated by the different models.
- pyroModel contain the model implemented using pyro primitives. This is used mainly for mcmc based inference.
- torchModel contain a point estimate version of the model (based on maximum likelyhood), it was not used in the final experiment.
- viModel contain the MAP point estimate version of the model and a mean field gaussian based version (for variational inference).

Then, a series of experiment scripts use an inference method to get the posterior. To provided a uniform interface for the next module in the pipeline, each of those scripts generate a pickle file containing samples from the posterior:

- mcmc_experiment generate those samples using a state of the art MCMC sampler from pyro.
- vi_experiment generate those samples using either the MAP point estimate model or the mean field gaussian model. Ensembling can be enable using a command line switch.

Finally, the results can be analysed by using the plots script.

The experiment and plotting scripts provide contextual help when called with the -h option:

	python module_name.py -h

## Citation

If you use our code in your project please cite our tutorial:

	@ARTICLE{9756596,
	author={Jospin, Laurent Valentin and Laga, Hamid and Boussaid, Farid and Buntine, Wray and Bennamoun, Mohammed},
	journal={IEEE Computational Intelligence Magazine}, 
	title={Hands-On Bayesian Neural Networks—A Tutorial for Deep Learning Users}, 
	year={2022},
	volume={17},
	number={2},
	pages={29-48},
	doi={10.1109/MCI.2022.3155327}
	}

