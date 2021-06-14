#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:07:19 2021

@author: laurent
"""

def model(coeffs, x, y) :
	"""
	Implementation of the paperfold model in numpy.
	"""
	
	branch1 = coeffs[...,0]*x + coeffs[...,1]*y + coeffs[...,2]
	branch2 = coeffs[...,3]*x + coeffs[...,4]*y + coeffs[...,5]
	
	return coeffs[...,6]*branch1 + coeffs[...,7]*(branch2*(branch2 > 0))