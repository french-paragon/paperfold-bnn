#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:55:43 2021

@author: laurent
"""

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import argparse as args

if __name__ == "__main__" :
	
	parser = args.ArgumentParser(description='Plot the samples and prediction uncertainty from a series of samples')
	
	parser.add_argument('input', default='./mcmc_samples.pkl', help="The location where the samples are stored")
	
	samples = pd.read_pickle(args.input)
	
	sns.pairplot(samples)
	
	plt.show()