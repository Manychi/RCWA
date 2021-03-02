import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy

PI = 3.14159265359

# Inputs
theta_inc = 0; lambda_0 = 20;   # Angle of incidence and wavelength of incident light
p = 5;                          # Grating period
alpha_l = 20; alpha_r = 30;     # Blazing and anti-blazing angle in degrees
d = 4;                        # Width of top cut
e0 = 1; e1 = 2;                 # Dielectric permitivity of surrounding medium and grating
N = 3;                         # Number of harmonics
Nl = 5;                         # Number of layers

# Geometry definition
alpha_lrad = alpha_l*PI/180; alpha_rrad = alpha_r*PI/180;   # Convert angles to radians
H  = (p-d)/(1/np.tan(alpha_lrad)+1/np.tan(alpha_rrad))      # Height of grating
Hs = H/Nl                                                   # Height of a slice
Sl = -p/2 + Hs/(2*np.tan(alpha_rrad))*(1+2*np.arange(0,Nl)) # Slice left boundary
Sr =  p/2 - Hs/(2*np.tan(alpha_lrad))*(1+2*np.arange(0,Nl)) # Slice right boundary
#print(np.arange(1,Nl+1))

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


#inputs for S-Matrix;
A = np.array([[1,1,3],[3,1,1],[1,3,1]]); #initiates the array, should load from script

#print(A)
#Build W, and Q for a loop of A, for each layer
Q, W = np.linalg.eig(A);                 # W = eigen vector, Q = eigen values

#Build X-matrix
k_0 = 2*PI/lambda_0
X = np.exp(-k_0*Q*Hs)  # X is now filled per layer for every

#initiate T matrix from W, X and Q
for i in range(N-1): #Provide range
    T_1     = np.array([[W[i+1],W[i+1]*X[i+1]], [W[i+1]*Q[1,i+1], -W[i+1]*Q[i+1]*X[i+1]]])
  #  T_1_inv = np.linalg.inv(T_1)
  #  T_2     = np.array([[ W[i]*X[i], W[i]],     [W[i]*Q[i]*X[i], -1*W[i]*Q[i]]])
    
#Build S-Matrix from given  total T matrix
#testje