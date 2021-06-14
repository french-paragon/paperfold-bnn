#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 09:59:31 2021

@author: laurent
"""


from dataset import getData
from viModel import VariationalInferenceModulePaperfold as paperfoldBnn
from viModel import PointEstimateModulePaperfold as paperfoldPe

import numpy as np
import pandas as pd

import torch
from torch.optim import Adam

import argparse as args

import time

if __name__ == "__main__" :
	"""
	Run the experiment for the VI model or the point estimate model.
	"""
	
	parser = args.ArgumentParser(description='Train the paperfold BNN using variational inference')
	
	parser.add_argument("--constrained", action="store_true", help="use the constrained model (8 data points instead of 4).")
	parser.add_argument("--pointestimate", action="store_true", help="use the point estimate model.")
	
	parser.add_argument('--samples', type=int, default = 1000, help="The number of sample to generate for each network")
	parser.add_argument('--trainsteps', type=int, default = 100, help="The number of training steps")
	parser.add_argument('--learningrate', type=float, default = 1e-3, help="The learning rate of the optimizer")
	parser.add_argument('--numnetworks', type=int, default = 1, help="The number of networks to train in parallel to perform ensembling")
	
	parser.add_argument('-o', '--output', default='./vi_samples.pkl', help="The location where to store the samples")
	
	args = parser.parse_args()
	
	x,y,z = getData(args.constrained)
	
	x = torch.from_numpy(x)[...,np.newaxis]
	y = torch.from_numpy(y)[...,np.newaxis]
	z = torch.from_numpy(z)[...,np.newaxis]
	
	modelList = []
	
	obs_sigma = 0.1
	lossbase = lambda z, zhat: torch.sum(torch.square((z - zhat)/obs_sigma))
	
	startTime = time.time()
	#train models
	for i in np.arange(args.numnetworks) :
	
		print("Train model {} out of {}".format(i+1, args.numnetworks))
		
		#create the model
		model = None
		if args.pointestimate :
			model = paperfoldPe(weight_sigma = 1., bias_sigma = 10.)
		else :
			model = paperfoldBnn(weight_sigma = 1., bias_sigma = 10.)
			
		optimizer = Adam(model.parameters(), lr=args.learningrate)
		optimizer.zero_grad()
		
		for s in np.arange(args.trainsteps) :
			
			zhat = model(x,y)
			
			l = lossbase(z, zhat) 
			l += model.evalAllLosses()
			
			optimizer.zero_grad()
			l.backward()
			
			optimizer.step()
			
			print("\r", "\tTrain step {}/{} Loss = {:.4f}".format(s+1, args.trainsteps, l.detach().cpu().item()), end="")
			
		print("")
			
		modelList.append(model)
	
	
	trainTime = time.time() - startTime
	print("Training time: {}s".format(trainTime))
		
	#sample models
	
	samples = None
	
	startTime = time.time()
	
	for i in np.arange(args.numnetworks) :
	
		print("Sample model {} out of {}".format(i+1, args.numnetworks))
	
		for s in np.arange(args.samples) :
			
			print("\r", "\tSample {}/{}".format(s+1, args.samples), end="")
		
			m = modelList[i]
			m(x,y) #call the network to generate a sample
			sample = m.getSampledParams()
			
			names = []
			flatSamples = None
			sites = sample.keys()
				
			for site in sites :
				
				s = sample[site].shape[-1]
				
				a = np.squeeze(sample[site])
				
				if flatSamples is None :
					flatSamples = a
				else :
					if (len(a.shape) == 0) :
						flatSamples = np.concatenate((flatSamples, a[np.newaxis]))
					else :
						flatSamples = np.concatenate((flatSamples, a))
					
				if s == 1 :
					names.append(site)
				else :
					names += ["{}.{}".format(site, i) for i in np.arange(1,s+1)]
		
			s = pd.DataFrame(data = flatSamples[np.newaxis,...], index=np.arange(1), columns=names)
			
			if samples is None :
				samples = s
			else :
				samples = samples.append(s, ignore_index=True)
			
		print("")
		
	executionTime = time.time() - startTime
	print("Time per sample: {}s".format(executionTime/(args.numnetworks*args.samples)))
				
	samples.to_pickle(args.output)