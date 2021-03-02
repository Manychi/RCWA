import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy
import cmath

PI = 3.14159265359

# imputs
N = 10;             #number of layers
kx_inc = 1;         #incoming kx vector
p = 1;              #grating period
k0 = 1;             #incoming wave constant
Nl = 25;            #number of layer
x = 1;              #distance
e_0 = 1;            
e_1 = 2;            
s_l = 1;            #right boundary
s_r = 1;            #left boundary
n= 10;              #number of harmonics

# make the Kx vector
k = np.empty(2*N+1);
size = np.prod(2*N+1)
matrixshape = (size, size);
Kx = np.zeros(matrixshape, dtype=np.cdouble);

for n in range(-N, N+1,):
    k[n+N] = (kx_inc - 2*PI*n/p)/k0;

np.fill_diagonal(Kx, k)

# make the E vector
e_r =e_0*p*(1j*(cmath.exp(-1j*PI*n*s_l/p)-cmath.exp(1j*PI*n))/(2*PI*n))+e_1*p*(1j*(cmath.exp(-1j*PI*n*s_r/p)-cmath.exp(1j*PI*n*s_l/p))/(2*PI*n))+e_0*p*(1j*(cmath.exp(-1j*PI*n)-cmath.exp(1j*PI*n*s_r/p))/(2*PI*n));
e = np.zeros(2*2*N+1, dtype=np.cdouble);
Ei = np.zeros(matrixshape, dtype=np.cdouble);

for m in range(-2*N, 2*N+1):
    e[m+2*N] = e_r*cmath.exp(2j*PI/p*m*x)

for i in range(0, 4*N+1):
    if i < 2*N:
        np.fill_diagonal(Ei[2*N-i:], e[i]);
    elif i == 2*N:
        np.fill_diagonal(Ei, e[i]);
    elif i > 2*N:
        np.fill_diagonal(Ei[:,i-2*N:], e[i]);
        
# diagonalize
Ai = np.linalg.matrix_power(Kx, 2) - Ei;
result = np.linalg.eig(Ai) # gives eigenvalues, eigenvectors