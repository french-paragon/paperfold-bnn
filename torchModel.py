#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 13:32:44 2021

@author: laurent
"""

import torch
import torch.nn as nn

class PointEstimateModulePaperfold(nn.Module) :
	"""
		PointEstimateModulePaperfold represent the point estimate weight BNN presented for the
		paperfold case study. 
		
		This version is just the plain ord maximum likelyhood version implemented with torch primitives.
		For our experiment we used the MAP version a a point estimate network, coded in the viModel module.
	"""
	
	def __init__(self) :
		super().__init__()
		
		self.layer1branch1 = nn.Linear(2,1)
		self.layer1branch2 = nn.Linear(2,1)
		
		self.layer2 = nn.Linear(2,1, bias = False)
		
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