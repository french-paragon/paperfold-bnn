#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:41:17 2021

@author: laurent
"""

import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from torch.distributions.normal import Normal


class VIModule(nn.Module) :
	"""
	A mixin class to attach loss functions to layer. This is usefull when doing variational inference with deep learning.
	"""
	
	def __init__(self, *args, **kwargs) :
		super().__init__(*args, **kwargs)
		
		self._internalLosses = []
		self.lossScaleFactor = 1
		
	def addLoss(self, func) :
		self._internalLosses.append(func)
		
	def evalLosses(self) :
		t_loss = 0
		
		for l in self._internalLosses :
			t_loss = t_loss + l(self)
			
		return t_loss
	
	def evalAllLosses(self) :
		
		t_loss = self.evalLosses()*self.lossScaleFactor
		
		for m in self.children() :
			if isinstance(m, VIModule) :
				t_loss = t_loss + m.evalAllLosses()*self.lossScaleFactor
				
		return t_loss

class L2RegularizedLinear(VIModule, nn.Linear) :
	"""
	A MAP linear layer with gaussian prior.
	"""
	
	def __init__(self, 
			  in_features, 
			  out_features,
			  bias=True, 
			  wPriorSigma = 1., 
			  bPriorSigma = 1.,
			  bias_init_cst = 0.0) :
		
		super().__init__(in_features, 
					   out_features,
					   bias=bias)
		
		if bias:
			self.bias.data.fill_(bias_init_cst)
		
		self.addLoss(lambda s : 0.5*s.weight.pow(2).sum()/wPriorSigma**2)
		
		if bias :
			
			self.addLoss(lambda s : 0.5*s.bias.pow(2).sum()/bPriorSigma**2)


class MeanFieldGaussianFeedForward(VIModule) :
	"""
	A feed forward layer with a Gaussian prior distribution and a Gaussian variational posterior.
	"""
	
	def __init__(self, 
			  in_features, 
			  out_features, 
			  bias = True,  
			  groups=1, 
			  weightPriorMean = 0, 
			  weightPriorSigma = 1.,
			  biasPriorMean = 0, 
			  biasPriorSigma = 1.,
			  initMeanZero = False,
			  initBiasMeanZero = False,
			  initPriorSigmaScale = 0.01) :
		
		
		super(MeanFieldGaussianFeedForward, self).__init__()
		
		self.samples = {'weights' : None, 'bias' : None, 'wNoiseState' : None, 'bNoiseState' : None}
		
		self.in_features = in_features
		self.out_features = out_features
		self.has_bias = bias
		
		self.weights_mean = Parameter((0. if initMeanZero else 1.)*(torch.rand(out_features, int(in_features/groups))-0.5))
		self.lweights_sigma = Parameter(torch.log(initPriorSigmaScale*weightPriorSigma*torch.ones(out_features, int(in_features/groups))))
			
		self.noiseSourceWeights = Normal(torch.zeros(out_features, int(in_features/groups)), 
								   torch.ones(out_features, int(in_features/groups)))
		
		#The prior and variational posterior contribution to the loss for the weight.
		self.addLoss(lambda s : 0.5*s.getSampledWeights().pow(2).sum()/weightPriorSigma**2)
		self.addLoss(lambda s : -self.out_features/2*np.log(2*np.pi) - 0.5*s.samples['wNoiseState'].pow(2).sum() - s.lweights_sigma.sum())
		
		if self.has_bias :
			self.bias_mean = Parameter((0. if initBiasMeanZero else 1.)*(torch.rand(out_features)-0.5))
			self.lbias_sigma = Parameter(torch.log(initPriorSigmaScale*biasPriorSigma*torch.ones(out_features)))
			
			self.noiseSourceBias = Normal(torch.zeros(out_features), torch.ones(out_features))
			
			#The prior and variational posterior contribution to the loss for the bias (added if bias are present).
			self.addLoss(lambda s : 0.5*s.getSampledBias().pow(2).sum()/biasPriorSigma**2)
			self.addLoss(lambda s : -self.out_features/2*np.log(2*np.pi) - 0.5*s.samples['bNoiseState'].pow(2).sum() - self.lbias_sigma.sum())
			
			
	def sampleTransform(self, stochastic=True) :
		self.samples['wNoiseState'] = self.noiseSourceWeights.sample().to(device=self.weights_mean.device)
		self.samples['weights'] = self.weights_mean + (torch.exp(self.lweights_sigma)*self.samples['wNoiseState'] if stochastic else 0)
		
		if self.has_bias :
			self.samples['bNoiseState'] = self.noiseSourceBias.sample().to(device=self.bias_mean.device)
			self.samples['bias'] = self.bias_mean + (torch.exp(self.lbias_sigma)*self.samples['bNoiseState'] if stochastic else 0)
		
	def getSampledWeights(self) :
		return self.samples['weights']
	
	def getSampledBias(self) :
		return self.samples['bias']
	
	def forward(self, x, stochastic=True) :
		
		self.sampleTransform(stochastic=stochastic)
		
		return nn.functional.linear(x, self.samples['weights'], bias = self.samples['bias'] if self.has_bias else None)
	

class VariationalInferenceModulePaperfold(VIModule) :
	"""
		VariationalInferenceModulePaperfold represent the stochastic weight BNN presented for the
		paperfold case study, implemented to learn the posterior using variational inference.
	"""
	
	def __init__(self, weight_sigma = 1., bias_sigma = 10.) :
		super().__init__()
		
		self.layer1branch1 = MeanFieldGaussianFeedForward(2,1, weightPriorSigma = weight_sigma, biasPriorSigma = bias_sigma)
		self.layer1branch2 = MeanFieldGaussianFeedForward(2,1, weightPriorSigma = weight_sigma, biasPriorSigma = bias_sigma)
		
		self.layer2 = MeanFieldGaussianFeedForward(2,1, bias = False, weightPriorSigma = weight_sigma)
		
	def getSampledParams(self) :
		
		return {'layer1branch1.weights' : self.layer1branch1.getSampledWeights().detach().cpu().numpy(),
			    'layer1branch1.bias' : self.layer1branch1.getSampledBias().detach().cpu().numpy(),
			    'layer1branch2.weights' : self.layer1branch2.getSampledWeights().detach().cpu().numpy(),
			    'layer1branch2.bias' : self.layer1branch2.getSampledBias().detach().cpu().numpy(),
			    'layer2.weights' : self.layer2.getSampledWeights().detach().cpu().numpy()}
		
	def forward(self, x, y, stochastic=True):
		
		d = torch.cat((x,y), -1)
		
		b1 = self.layer1branch1(d, stochastic=stochastic)
		b2 = nn.functional.relu(self.layer1branch2(d, stochastic=stochastic))
		
		d = torch.cat((b1,b2), -1)
		
		sample = self.layer2(d, stochastic=stochastic)
		return sample
	

class PointEstimateModulePaperfold(VIModule) :
	"""
		PointEstimateModulePaperfold represent the MAP point estimate weight BNN presented for the
		paperfold case study, implemented to learn a point estimate.
	"""
	
	def __init__(self, weight_sigma = 1., bias_sigma = 10.) :
		super().__init__()
		
		self.layer1branch1 = L2RegularizedLinear(2,1, 
			  wPriorSigma = weight_sigma, 
			  bPriorSigma = bias_sigma,
			  bias_init_cst = 0.5)
		self.layer1branch2 = L2RegularizedLinear(2,1, 
			  wPriorSigma = weight_sigma, 
			  bPriorSigma = bias_sigma)
		
		self.layer2 = L2RegularizedLinear(2,1, bias = False, 
			  wPriorSigma = weight_sigma, 
			  bPriorSigma = bias_sigma)
		
	def getSampledParams(self) :
		
		return {'layer1branch1.weights' : self.layer1branch1.weight.detach().cpu().numpy(),
			    'layer1branch1.bias' : self.layer1branch1.bias.detach().cpu().numpy(),
			    'layer1branch2.weights' : self.layer1branch2.weight.detach().cpu().numpy(),
			    'layer1branch2.bias' : self.layer1branch2.bias.detach().cpu().numpy(),
			    'layer2.weights' : self.layer2.weight.detach().cpu().numpy()}
		
	def forward(self, x, y):
		
		d = torch.cat((x,y), -1)
		
		b1 = self.layer1branch1(d)
		b2 = nn.functional.relu(self.layer1branch2(d))
		
		d = torch.cat((b1,b2), -1)
		
		sample = self.layer2(d)
		return sample