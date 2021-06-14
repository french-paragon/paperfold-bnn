#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 11:27:22 2021

@author: laurent
"""

from dataset import getData
from pyroModel import WeightRegressionModulePaperfold as paperfoldBnn

import numpy as np
import pandas as pd

import torch

from pyro.infer.mcmc import MCMC, HMC, NUTS

import argparse as args

import time

if __name__ == "__main__" :
	"""
	Run the experiment for the MCMC model
	"""
	
	parser = args.ArgumentParser(description='Train the paperfold BNN using an MCMC sampler')
	
	parser.add_argument("--constrained", action="store_true", help="use the constrained model.")
	
	parser.add_argument('--method', choices=['nuts', 'hmc'], default = 'nuts', help="The mcmc method to use for training")
	parser.add_argument('--samples', type=int, default = 1000, help="The number of samples each chain will generate")
	parser.add_argument('--warmup', type=int, default = 100, help="The number of warmup steps in each chains")
	parser.add_argument('--numchains', type=int, default = 1, help="The number of chains to run")
	
	parser.add_argument('-o', '--output', default='./mcmc_samples.pkl', help="The location where to store the samples")
	
	args = parser.parse_args()
	
	x,y,z = getData(args.constrained)
	
	x = torch.from_numpy(x)[...,np.newaxis]
	y = torch.from_numpy(y)[...,np.newaxis]
	z = torch.from_numpy(z)[...,np.newaxis]
	
	#create the model, should be a callable containing pyro primitives
	model = paperfoldBnn(weight_sigma = 1., bias_sigma = 10., obs_sigma = 0.1)
	
	hmc_kernel = None
	if args.method == 'hmc' :
		hmc_kernel = HMC(model, step_size=0.0855, num_steps=4)
	else :
		hmc_kernel = NUTS(model, adapt_step_size=True)
	
	chain = MCMC(hmc_kernel, num_samples=args.samples, num_chains = args.numchains, warmup_steps=args.warmup)
		
	startTime = time.time()
	
	chain.run(x,y,z)
	
	duration = time.time() - startTime
	print("Total time: {}s".format(duration))
	print("Time per sample: {}s".format(duration/((args.warmup+args.samples)*args.numchains)))
	
	samples = chain.get_samples()
	sites = samples.keys()
	
	flatSamples = None
	names = []
	
	for site in sites :
		
		s = samples[site].shape[-1]
		
		a = samples[site].squeeze().numpy()
		if len(a.shape) < 2 :
			a = a[...,np.newaxis]
		
		if flatSamples is None :
			flatSamples = a
		else :
			flatSamples = np.concatenate((flatSamples, a),1)
			
		if s == 1 :
			names.append(site)
		else :
			names += ["{}.{}".format(site, i) for i in np.arange(1,s+1)]
	
	samples = pd.DataFrame(data = flatSamples, index=np.arange(args.samples*args.numchains), columns=names)
	samples.to_pickle(args.output) #save samples