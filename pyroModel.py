#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 10:38:04 2021

@author: laurent
"""

import numpy as np

import torch
from torch import nn

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.nn import PyroSample

class LinearBayesianWeights(PyroModule[nn.Linear]) :
	"""
		The LinearBayesianWeights is a usual feedforward linear layer, 
		but where the weight and bias are replaced by a stochastic variable with a prior
	"""
	
	def __init__(self, inFeatures, outFeatures, bias = True, weight_sigma = 1., bias_sigma = 10.) :
		
		super().__init__(inFeatures, outFeatures, bias)
		#PyroSample is used to indicate that any acess to a given PyroModule parameter should be replaced by a pyro.sample call.
		self.weight = PyroSample(dist.Normal(0., weight_sigma).expand([outFeatures, inFeatures]).to_event(2))
		if bias :
			self.bias = PyroSample(dist.Normal(0., bias_sigma).expand([outFeatures]).to_event(1))
		
	
class WeightRegressionModulePaperfold(PyroModule) :
	"""
		WeightRegressionModulePaperfold represent the stochastic weight BNN presented for the
		paperfold case study.
	"""
	
	def __init__(self, weight_sigma = 1., bias_sigma = 10., obs_sigma = 0.1) :
		super().__init__()
		
		self.layer1branch1 = LinearBayesianWeights(2,1, weight_sigma = weight_sigma, bias_sigma = bias_sigma)
		self.layer1branch2 = LinearBayesianWeights(2,1, weight_sigma = weight_sigma, bias_sigma = bias_sigma)
		
		self.layer2 = LinearBayesianWeights(2,1, bias = False, weight_sigma = weight_sigma)
		self.obs_sigma = obs_sigma
		
	def forward(self, x, y, z=None):
		
		d = torch.cat((x,y), -1)
		
		b1 = self.layer1branch1(d)
		b2 = nn.functional.relu(self.layer1branch2(d))
		
		d = torch.cat((b1,b2), -1)
		
		mean = self.layer2(d)
		# A pyro plate is used to indicate a PGM plate.
		# It can be exploited by different pyro algorithm to exploit independance and speed up computations..
		with pyro.plate("data", z.shape[0]):
			# The pyro sample call return a random sample from a distribution, 
			# but it also leave some informations in a global dictionary to give some informations about the actual sample probability.
			pyro.sample("obs", dist.Normal(mean, self.obs_sigma), obs=z)
		return mean