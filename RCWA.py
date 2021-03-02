# testing GIT2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy

PI = 3.14159265359

# Inputs
theta_inc = 0; lambda_0 = 20;   # Angle of incidence and wavelength of incident light
p = 1;                          # Grating period
alpha_l = 30; alpha_r = 50;     # Blazing and anti-blazing angle in degrees
d = 0.5;                        # Width of top cut
e0 = 1; e1 = 2;                 # Dielectric permitivity of surrounding medium and grating
N = 100;                         # Number of harmonics
Nl = 50;                         # Number of layers

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
            result[n+2*N] = 0                               # In case n is equal to zero do something??
    return result                                           # Return the resulting array

# Function to get the Kx2 and Earrays in a matrix form
def ArraysToA(Earray,Kx2):
    A = np.zeros((2*N+1,2*N+1), dtype=np.cdouble)
    for i in range(0, 4*N+1):
        if i < 2*N:
            np.fill_diagonal(A[2*N-i:], -Earray[i]);
        elif i == 2*N:
            np.fill_diagonal(A, Kx2[i]-Earray[i]);
        else:
            np.fill_diagonal(A[:,i-2*N:], -Earray[i]);
    return A  # Return the A matrix for that layer as defined in the slides

# Find the Ai matrix for all layers i
A = np.zeros((2*N+1,2*N+1,Nl), dtype=np.cdouble)            # Memory allocation
for i in range(Nl):                                         # For every layer
    Earray = e0*integral(-p/2,Sl[i])+e1*integral(Sl[i],Sr[i])+e0*integral(Sr[i],p/2) # Calculate the complete integral for the three regions
    A[:,:,i] = ArraysToA(Earray,Kx2)                        # Convert to a matrix form where last index is the layer number



