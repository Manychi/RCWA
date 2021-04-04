import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import numpy as np
from functools import reduce # only in Python 3
import RCWA 

'''INPUTS'''
theta_inc = 10; lambda_0 = 0.4;  # Angle of incidence and wavelength of incident light
p = 1;                           # Grating period
alpha_l = 20.; alpha_r = 40.;    # Blazing and anti-blazing angle in degrees
d = 0.5;                         # Width of top cut
e0 = 1; e1 = 2;  e2 =1;          # Dielectric permitivity of surrounding medium and grating

N = 10;                           # Number of harmonics
Nl = 5;                          # Number of layers excluding sub- and superstrate layers
Nlt = Nl+2;                      # Total number of layers including sub- and superstrate layers

accuracy = 100                   # Accuracy to determine the steps in the plots

RCWA.solver(theta_inc,lambda_0,p,alpha_l,alpha_r, d,e0,e1,e2,N,Nl,accuracy)