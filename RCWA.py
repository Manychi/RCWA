# testing GIT2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import scipy as sp

PI = 3.14159265359

# Inputs
theta_inc = 0; lambda_0 = 20;   # Angle of incidence and wavelength of incident light
p = 50;                          # Grating period
alpha_l = 30; alpha_r = 50;     # Blazing and anti-blazing angle in degrees
d = 0.5;                        # Width of top cut
e0 = 1; e1 = 2;                 # Dielectric permitivity of surrounding medium and grating
N = 10;                         # Number of harmonics
Nl = 10;                         # Number of layers

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

A = np.zeros((2*N+1,2*N+1,Nl), dtype=np.cdouble)            # Memory allocation
for i in range(Nl):                                         # For every layer
    Earray = e0*integral(-p/2,Sl[i])+e1*e0*integral(Sl[i],Sr[i])+e0*integral(Sr[i],p/2) # Calculate the complete integral for the three regions
    A[:,:,i] = ArraysToA(Earray,Kx2)                        # Convert to a matrix form where last index is the layer number

# Find eigen vector,W, and values,Q, for each layer
Q = np.zeros((2*N+1,Nl),      dtype = np.cdouble)     #Memory allocation for Q
W = np.zeros((2*N+1,2*N+1,Nl),dtype = np.cdouble)     #Memory allocation for W
X = Q               # Memory allocation for X    
                                  
for i in range(Nl): # For every layer 
    Q[:,i], W[:,:,i] = np.linalg.eig(A[:,:,i]) # Computes eigen value and vector for every layer matrix A
 
#Build X-matrix
X = np.exp(-k_0*Q*Hs)

  # X is now filled per layer for Q

# T matrix solution
 
#memory allocation

T_full = np.zeros((2*N+1, 2*N+1, Nl), dtype=np.cdouble)
X_full = np.zeros((2*N+1, 2*N+1), dtype=np.cdouble)
T_Matrix_2_intermediate = np.zeros((2*N+1, 2*N+1,Nl),dtype=np.cdouble)
for i in range(Nl-1):
     
    #Different vector initialization
    T11 = W[:,:,i+1]            #proceeding layer W
    T12 = W[:,:,i]              #current layer W
    T21 = np.matmul(W[:,:,i+1],Q[:,i+1])   #proceeding layer W and Q
    T22 = np.matmul(W[:,:,i],Q[:,i])      #current layer W and Q
   
    #Parts of T to find T total for each layer
    # T_Matrix_1 = np.array([[np.identity(Nl),0],[0,np.linalg.inv(np.diag(X[:,i+1]))]])
    # T_Matrix_2_intermediate = np.array([[T11,T11],[T21,-T21]])
    # T_Matrix_2 = np.linalg.inv(T_Matrix_2_intermediate)
    # T_Matrix_3 = np.array([[T12,T12],[T22,-T22]])
    # T_Matrix_4 = np.array([[X[:,i],0],[0,np.identity(Nl)]])
    
    # T_M2_hand  = 1/(T11*-T21-T11*T21)*np.array([[T11,-T11],[-T21,-T21]])
    # T_full     = np.matmul(np.matmul(T_Matrix_1,T_Matrix_2),np.matmul(T_Matrix_3,T_Matrix_4))
    
    T_Part_1 = np.linalg.inv(np.array([[T11, np.matmul(T11,X[:,i+1])], 
                                        [T21, np.matmul(-T21,X[:,i+1])]]))
    T_Part_2 = np.array([[np.matmul(T12,X[:,i]),T12],[np.matmul(T12,X[:,i]),T12]])
    T_full   = np.multiply(T_Part_1,T_Part_2)
    
    #*T_Matrix_3*T_Matrix_4
            
#Build S-Matrix from given  total T matrix

