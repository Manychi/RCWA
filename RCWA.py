# testing GIT2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from functools import reduce # only in Python 3
import scipy as sp

PI = 3.14159265359

# Inputs
theta_inc = 0; lambda_0 = 20;   # Angle of incidence and wavelength of incident light
p = 50;                          # Grating period
alpha_l = 30; alpha_r = 50;     # Blazing and anti-blazing angle in degrees
d = 0.5;                        # Width of top cut
e0 = 1; e1 = 2;                 # Dielectric permitivity of surrounding medium and grating
N = 10;                         # Number of harmonics
Nl = 10;                        # Number of layers excluding sub- and superstrate layers
Nlt = Nl+2;                     # Total number of layers including sub- and superstrate layers

# Geometry definition
alpha_lrad = alpha_l*PI/180; alpha_rrad = alpha_r*PI/180;   # Convert angles to radians
H  = (p-d)/(1/np.tan(alpha_lrad)+1/np.tan(alpha_rrad))      # Height of grating
Hs = H/Nl                                                   # Height of a slice
Sl = -p/2 + Hs/(2*np.tan(alpha_rrad))*(1+2*np.arange(0,Nl)) # Slice left boundary
Sr =  p/2 - Hs/(2*np.tan(alpha_lrad))*(1+2*np.arange(0,Nl)) # Slice right boundary

# Draw Geometry
figure, ax = plt.subplots(1)    # Make a plot

grating = [[-p/2,-H], [p/2,-H], # Make an array consiting of the corner coordinates of the grating
           [p/2-H/np.tan(alpha_lrad),0],[-p/2+H/np.tan(alpha_rrad),0]]
grating.append(grating[0])      # Needed to close the contour of the grating
x, y = zip(*grating)            # Get x and y coordinates separately in a tuple
plt.plot(x,y)                   # Plot the grating shape

for i in range(Nl):             # Draw the rectangular slices
    rect = patches.Rectangle((Sl[i], Hs*i-H), -Sl[i] + Sr[i], Hs,edgecolor='r',facecolor="none")
    ax.add_patch(rect)
    
plt.show()                      # Show plot

# Find Kx2 array
k_0 = 2*PI/lambda_0                 # Find k0 from inputs
kinc_x = k_0*np.cos(theta_inc)      # Find the x component from the incident wave
n = np.arange(-N,N+1)               # Define the range of n
Kx = kinc_x-2*PI*n/p                # 
Kx2 = Kx*Kx/(k_0*k_0)               # Find the Kx squared matrix diagonal

# Function to solve the integral for all harmonics and put it in an array
def integral(boundaryl,boundaryr):                          # Inputs are the left and right boundary
    result = np.zeros((4*N+1,1), dtype=np.cdouble);         # Memory allocation
    for n in range(-2*N,2*N+1):                             # Do the calculation for all harmonics present in the E-matrix
        if n != 0:                                          # n equal to zero gives devision by 0 so must be taken separately
            result[n+2*N] = np.exp(-2j*PI*n*boundaryl/p)*p*1j/(2*PI*n)
            - np.exp(-2j*PI*n*boundaryr/p)*p*1j/(2*PI*n)    # Calculating the integral
        else:
            result[n+2*N] = boundaryl -boundaryr            # In case n is equal to zero integral is over a constant
    return result                                           # Return the resulting array

# Function to get the Kx2 and Earrays in a matrix form
def ArraysToA(Earray,Kx2):
    A = np.zeros((2*N+1,2*N+1), dtype=np.cdouble)
    for i in range(0, 4*N+1):
        if i < 2*N:
            np.fill_diagonal(A[2*N-i:], -Earray[i]);
        elif i == 2*N:
            np.fill_diagonal(A, Kx2-Earray[i]);
        else:
            np.fill_diagonal(A[:,i-2*N:], -Earray[i]);
    return A  # Return the A matrix for that layer as defined in the slides

# Find the Ai matrix for all layers i
A = np.zeros((2*N+1,2*N+1,Nlt), dtype=np.cdouble)             # Memory allocation
A[:,:,0] = ArraysToA(e0*integral(-p/2,p/2),Kx2)               # Determine A in the superstrate
A[:,:,Nlt-1] = ArraysToA(e1*integral(-p/2,p/2),Kx2)           # Determine A in the substrate

for i in range(Nl):                                           # For every layer in the geometry
    Earray = e0*integral(-p/2,Sl[i])+e1*integral(Sl[i],Sr[i])+e0*integral(Sr[i],p/2) # Calculate the complete integral for the three regions
    A[:,:,i+1] = ArraysToA(Earray,Kx2)                        # Convert to a matrix form where last index is the layer number

# Find eigen vector,W, and values,Q, for each layer
Q = np.zeros((2*N+1,Nl),      dtype = np.cdouble)     #Memory allocation for Q
W = np.zeros((2*N+1,2*N+1,Nl),dtype = np.cdouble)     #Memory allocation for W
X = Q               # Memory allocation for X    
                                  
for i in range(Nl): # For every layer 
    Q[:,i], W[:,:,i] = np.linalg.eig(A[:,:,i]) # Computes eigen value and vector for every layer matrix A
 
#Build X-matrix
X = np.exp(-k_0*Q*Hs)

Tsubi = np.zeros((4*N+2, 4*N+2), dtype=np.cdouble)
Tsubi1 = np.zeros((4*N+2, 4*N+2), dtype=np.cdouble)

def Tmatrix(Wi,Qi,Wi1,Qi1):
    for i in range(2*N+1):
        for j in range(2*N+1):
            Tsubi[i,j] = Wi[i,j]
            Tsubi[i,j+2*N+1] = Wi[i,j]
            Tsubi[i+2*N+1,j] = Wi[i,j]*Qi[i]
            Tsubi[i+2*N+1,j+2*N+1] = -Wi[i,j]*Qi[i]
    
            Tsubi1[i,j] = Wi1[i,j]
            Tsubi1[i,j+2*N+1] = Wi1[i,j]
            Tsubi1[i+2*N+1,j] = Wi1[i,j]*Qi1[i]
            Tsubi1[i+2*N+1,j+2*N+1] = -Wi1[i,j]*Qi1[i]
            
    return np.matmul(np.linalg.inv(Tsubi1),Tsubi)

def TtoS(T,Qi,Qi1):
    T11 = T[0:2*N+1,0:2*N+1]
    T12 = T[2*N+1:4*N+2,0:2*N+1]
    T21 = T[0:2*N+1,2*N+1:4*N+2]
    T22 = T[2*N+1:4*N+2,2*N+1:4*N+2]
    
    Xi = np.diag(np.exp(-k_0*Qi*Hs))
    Xi1 = np.diag(np.exp(-k_0*Qi1*Hs))
    
    S11 = np.matmul(T11,Xi) - reduce(np.matmul,[T12,np.linalg.inv(T22),T21,Xi])
    S12 = reduce(np.matmul,[T12,np.linalg.inv(T22),Xi1])
    S21 = -reduce(np.matmul,[np.linalg.inv(T22),T21,Xi])
    S22 = np.matmul(np.linalg.inv(T22),Xi1)
    return S11, S12, S21, S22

# def Redheffer(S11i, S12i, S21i, S22i, S11i1, S12i1, S21i1, S22i1):
#     S11 = np.matmul(S11i1,inv(np.identity(2*N+1))-np.matmul(S12i,S))
    
    
T1 = np.identity(4*N+2,dtype = np.cdouble);
Qi, Wi = np.linalg.eig(A[:,:,i]) # Computes eigen value and vector for every layer matrix A


def EVisual(r,t,c_plus,c_min,Nlt):
    kz_n = np.sqrt(k_0^2*e0 - Kx^2)
    z = np.arange(start = 0, stop = H, step = H/Nl)
    E_vis = np.zeros((2*N+1,2*N+1,Nlt), dtype=np.cdouble)   
    for i in range(Nlt):
        if i == 0:
            E_vis[:,:,i] = (t*np.exp(-1j*kz_n*z) + r*np.exp(1j*kz_n*z))
                            
        if i == Nlt:
            E_vis[:,:,i] = (t*np.exp(-1j*kz_n*(z[i]))*np.exp(-1j*Kx*x))  
            
        else:
            E_vis[:,:,i] = W[i,i]*(np.exp(-k_0*Q[i]*z[i-1])*c_plus + np.exp(k_0*Q[i]*z[i])*c_min)*np.exp(-1j*Kx*x)
        
    return E_vis


for i in range(Nl-1):
    Qi1, Wi1 = np.linalg.eig(A[:,:,i+1]) # Computes eigen value and vector for every layer matrix A
    T = Tmatrix(W[:,:,i],Q[:,i],W[:,:,i+1],Q[:,i+1])
    S11, S12, S21, S22 = TtoS(T,Qi,Qi1)

    Qi=Qi1
    Wi=Wi1
    
    
    
         
         
         

