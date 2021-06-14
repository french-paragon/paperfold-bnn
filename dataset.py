#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:56:51 2021

@author: laurent
"""

import numpy as np

import matplotlib.pyplot as plt

def getData(constrained : bool = False) :
	"""
	Generate the data as described in figure 19 of the paper.
	"""
	
	x = np.array([np.sqrt(2), 0, -np.sqrt(2), 0], dtype=np.float32)
	y = np.array([0, np.sqrt(2), 0, -np.sqrt(2)], dtype=np.float32)
	z = np.array([1, 0, 1, 0], dtype=np.float32)
	
	if constrained :
		x = np.append(x, np.sqrt(1/2)*np.array([1, 1, -1, -1], dtype=np.float32))
		y = np.append(y, np.sqrt(1/2)*np.array([1, -1, 1, -1], dtype=np.float32))
		z = np.append(z, np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32))
		
	return x,y,z
	
def getZNoise(sigma : float = 0.1, constrained : bool = False) :
	"""
	Generate a noise sample to apply on top of the data.
	"""
	
	n = 8 if constrained else 4
	return np.random.default_rng().normal(0, sigma, n).astype(np.float32)



if __name__ == ("__main__") :
	"""
	If the file is called as a script just do a 3D scatter plot of the data.
	"""
	
	x,y,z = getData(True)
	
	fig = plt.figure("constrained data")
	ax = fig.add_subplot(projection='3d')
	ax.scatter(x,y,z, marker = 'o')
	ax.set_box_aspect([1,1,0.5])
	
	plt.show()