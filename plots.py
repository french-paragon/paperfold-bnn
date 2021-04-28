#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:55:43 2021

@author: laurent
"""

from numpyModel import model as npmodel
import numpy as np

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import argparse as args

def plotLineRestrictions(numpy_samples, x, y) :
	
	graphyeq0 = npmodel(numpy_samples, x, 0)
	graphxeq0 = npmodel(numpy_samples, 0, y)
	graphxeqy = npmodel(numpy_samples, x, x)
	graphxeqmy = npmodel(numpy_samples, y, -y)
	
	fig, axs = plt.subplots(2,2)
	fig.canvas.set_window_title('Model predictions across different axis')
	
	axs[0,0].plot(x, graphyeq0, linestyle='', marker='o', color='blue')
	axs[0,0].set_title('y=0')
	
	axs[0,1].plot(y, graphxeq0, linestyle='', marker='o', color='orange')
	axs[0,1].set_title('x=0')
	
	axs[1,0].plot(x, graphxeqy, linestyle='', marker='o', color='red')
	axs[1,0].set_title('x=y')
	
	axs[1,1].plot(y, graphxeqmy, linestyle='', marker='o', color='green')
	axs[1,1].set_title('x=-y')
	
	return fig, axs

def graphMoments(numpy_samples, extent = (-np.sqrt(2), np.sqrt(2), -np.sqrt(2), np.sqrt(2)), res = 200) :
	
	xs = np.linspace(extent[0], extent[1], res)
	ys = np.linspace(extent[2], extent[3], res)
	
	xx, yy = np.meshgrid(xs, ys)
	
	graph = npmodel(numpy_samples[np.newaxis, np.newaxis, ...], xx[... , np.newaxis], yy[... , np.newaxis])
	
	mean = np.mean(graph, axis=-1)
	std = np.std(graph, axis=-1)
	
	figMean = plt.figure('Mean')
	ax = figMean.add_subplot(1,1,1)
	img = ax.imshow(mean, origin = 'lower', extent=extent)
	plt.xlabel('x')
	plt.ylabel('y')
	clb = plt.colorbar(img)
	clb.set_label("z")
	
	figStd = plt.figure('Standard deviation')
	ax = figStd.add_subplot(1,1,1)
	img = plt.imshow(std, origin = 'lower', extent=extent)
	plt.xlabel('x')
	plt.ylabel('y')
	clb = plt.colorbar(img)
	clb.set_label("σ_z")
	
	return figMean, figStd
	

if __name__ == "__main__" :
	
	parser = args.ArgumentParser(description='Plot the samples and prediction uncertainty from a series of samples')
	
	parser.add_argument('input', help="The location where the samples are stored")
	
	args = parser.parse_args()
	
	samples = pd.read_pickle(args.input)
	
	sns.pairplot(samples)
	
	nsamples = samples.shape[0]
	
	x = np.random.rand(nsamples)*2*np.sqrt(2) - np.sqrt(2)
	y = np.random.rand(nsamples)*2*np.sqrt(2) - np.sqrt(2)
	
	numpy_samples = samples.to_numpy()
	
	plotLineRestrictions(numpy_samples, x, y)
	
	graphMoments(numpy_samples)
	
	plt.show()