import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from functools import reduce # only in Python 3

PI = 3.14159265359

# Inputs
theta_inc = 30; lambda_0 = 2;   # Angle of incidence and wavelength of incident light
p = 1;                          # Grating period
alpha_l = 90; alpha_r = 30;     # Blazing and anti-blazing angle in degrees
d = 0.5;                        # Width of top cut
e0 = 1.2; e1 = 2;  e2 =1.1;     # Dielectric permitivity of surrounding medium and grating

N = 1;                         # Number of harmonics
Nl = 1;                        # Number of layers excluding sub- and superstrate layers
Nlt = Nl+2;                     # Total number of layers including sub- and superstrate layers

# Geometry definition
alpha_lrad = alpha_l*PI/180; alpha_rrad = alpha_r*PI/180;   # Convert angles to radians
H  = (p-d)/(1/np.tan(alpha_lrad)+1/np.tan(alpha_rrad))      # Height of grating
Hs = H/Nl                                                   # Height of a slice
Sl = -p/2 + Hs/(2*np.tan(alpha_lrad))*(1+2*np.arange(0,Nl)) # Slice left boundary
Sr =  p/2 - Hs/(2*np.tan(alpha_rrad))*(1+2*np.arange(0,Nl)) # Slice right boundary

# Draw Geometry
figure, ax = plt.subplots(1)    # Make a plot

grating = [[-p/2,-H], [p/2,-H], # Make an array consiting of the corner coordinates of the grating
           [p/2-H/np.tan(alpha_rrad),0],[-p/2+H/np.tan(alpha_lrad),0]]
grating.append(grating[0])      # Needed to close the contour of the grating
x, y = zip(*grating)            # Get x and y coordinates separately in a tuple
plt.plot(x,y)                   # Plot the grating shape

for i in range(Nl):             # Draw the rectangular slices
    rect = patches.Rectangle((Sl[i], Hs*i-H), -Sl[i] + Sr[i], Hs,edgecolor='r',facecolor="none")
    ax.add_patch(rect)
    
plt.show()                      # Show plot

# Find Kx2 array
k_0 = 2*PI/lambda_0                 # Find k0 from inputs
k_0=1
kinc_x = k_0*np.cos(theta_inc)      # Find the x component from the incident wave
n = np.arange(-N,N+1)               # Define the range of n
Kx = kinc_x-2*PI*n/p                # 
Kx2 = Kx*Kx/(k_0*k_0)               # Find the Kx squared matrix diagonal

# Function to solve the integral for all harmonics and put it in an array
def integral(boundaryl,boundaryr):                            # Inputs are the left and right boundary
    result = np.zeros((4*N+1,1), dtype=np.cdouble);           # Memory allocation
    for n in range(-2*N,2*N+1):                               # Do the calculation for all harmonics present in the E-matrix
        if n != 0:                                            # n equal to zero gives devision by 0 so must be taken separately
            result[n+2*N] = np.exp(-2j*PI*n*boundaryl/p)*p*1j/(2*PI*n)
            - np.exp(-2j*PI*n*boundaryr/p)*p*1j/(2*PI*n)      # Calculating the integral
        else:
            result[n+2*N] = boundaryl -boundaryr              # In case n is equal to zero integral is over a constant
    return result                                             # Return the resulting array

# Function to get the Kx2 and Earrays in a matrix form
def ArraysToA(Earray,Kx2):
    A = np.zeros((2*N+1,2*N+1), dtype=np.cdouble)             # Memory allocation
    for i in range(4*N+1):                                    # For all harmonics 
        if i < 2*N:
            np.fill_diagonal(A[2*N-i:], -Earray[i]);
        elif i == 2*N:
            np.fill_diagonal(A, Kx2-Earray[i]);
        else:
            np.fill_diagonal(A[:,i-2*N:], -Earray[i]);
    return A                                                  # Return the A matrix for that layer as defined in the slides

# Find the Ai matrix for all layers i
A = np.zeros((2*N+1,2*N+1,Nlt), dtype=np.cdouble)             # Memory allocation
A[:,:,0] = ArraysToA(e0*integral(-p/2,p/2),Kx2)               # Determine A in the superstrate
A[:,:,Nlt-1] = ArraysToA(e2*integral(-p/2,p/2),Kx2)           # Determine A in the substrate

for i in range(Nl):                                           # For every layer in the geometry
    Earray = e0*integral(-p/2,Sl[i])+e1*integral(Sl[i],Sr[i])+e0*integral(Sr[i],p/2) # Calculate the complete integral for the three regions
    A[:,:,i+1] = ArraysToA(Earray,Kx2)                        # Convert to a matrix form where last index is the layer number

# Function to find the Tmatrix of interface of layer i and i+1 based on the eigenvalues Q and eigenvectors W
def Tmatrix(Wi,Qi,Wi1,Qi1):

    Tsubi = np.zeros((4*N+2, 4*N+2), dtype=np.cdouble)        # Memory allocation
    Tsubi1 = np.zeros((4*N+2, 4*N+2), dtype=np.cdouble)       # Memory allocation
    
    for i in range(2*N+1):                                    # For all harmonics horizontal
        for j in range(2*N+1):                                # For all harmonics vertical
            Tsubi[i,j] = Wi[i,j]                              # Fill in part of T11 for layer i
            Tsubi[i,j+2*N+1] = Wi[i,j]                        # Fill in part of T12 for layer i
            Tsubi[i+2*N+1,j] = Wi[i,j]*Qi[i]                  # Fill in part of T21 for layer i
            Tsubi[i+2*N+1,j+2*N+1] = -Wi[i,j]*Qi[i]           # Fill in part of T22 for layer i
            
            Tsubi1[i,j] = Wi1[i,j]                            # Fill in part of T11 for layer i+1
            Tsubi1[i,j+2*N+1] = Wi1[i,j]                      # Fill in part of T11 for layer i+1
            Tsubi1[i+2*N+1,j] = Wi1[i,j]*Qi1[i]               # Fill in part of T11 for layer i+1
            Tsubi1[i+2*N+1,j+2*N+1] = -Wi1[i,j]*Qi1[i]        # Fill in part of T11 for layer i+1
    return np.matmul(np.linalg.inv(Tsubi1),Tsubi)             # Return the Tmatrix of Ti,i+1

# Function to convert Ti,i+1 to Si,i+1 based on the T matrix and the eigenvalues
def TtoS(T,Qi,Qi1):
    # Split the T matrix into the four subquadrants
    T11 = T[0:2*N+1,0:2*N+1]
    T12 = T[2*N+1:4*N+2,0:2*N+1]
    T21 = T[0:2*N+1,2*N+1:4*N+2]
    T22 = T[2*N+1:4*N+2,2*N+1:4*N+2]
    
    # Compute the X vectors of layer i and i+1
    Xi = np.diag(np.exp(-k_0*Qi*Hs))
    Xi1 = np.diag(np.exp(-k_0*Qi1*Hs))
    
    # Define the four quadrants of the S-matrix
    S11 = np.matmul(T11,Xi) - reduce(np.matmul,[T12,np.linalg.inv(T22),T21,Xi])
    S12 = reduce(np.matmul,[T12,np.linalg.inv(T22),Xi1])
    S21 = -reduce(np.matmul,[np.linalg.inv(T22),T21,Xi])
    S22 = np.matmul(np.linalg.inv(T22),Xi1)
    return np.array([S11, S12, S21, S22])                             # Return all the quadrants of the computed S-matrix


# Function to calculate the Redheffer star product from all quadrants of S1,i and Si,i+1
# The indeces between brackets of the S-matrices correspond to S11, S12, S21, S22 respectively
def Redheffer(Sglobal_old,S):  #Sglobal is the total S matrix, S is the next layer
    Sglobal_new11 = reduce(np.matmul, [S[0], np.linalg.inv(np.identity(2*N+1)-np.matmul(Sglobal_old[1], S[2])) ,Sglobal_old[0]]) 
    Sglobal_new12 = np.add(S[1], reduce(np.matmul,[S[0], Sglobal_old[1], np.linalg.inv(np.identity(2*N+1)-np.matmul(S[2], Sglobal_old[1])) ,S[3]])) 
    Sglobal_new21 = np.add(Sglobal_old[2], reduce(np.matmul,[Sglobal_old[3], S[2], np.linalg.inv(np.identity(2*N+1)-np.matmul(Sglobal_old[1], S[2])) ,Sglobal_old[0]])) 
    Sglobal_new22 = reduce(np.matmul, [Sglobal_old[3], np.linalg.inv(np.identity(2*N+1)-np.matmul(S[2], Sglobal_old[1])) ,S[3]])  
    return np.array([Sglobal_new11, Sglobal_new12, Sglobal_new21, Sglobal_new22])                  # Return all the quadrants of the computed global S-matrix

# Calculate the eigenvalues and eigenvectors for every layer
Q2 = np.zeros((2*N+1), dtype=np.cdouble)                    # Memory allocation
W  = np.zeros((2*N+1, 2*N+1,Nlt), dtype=np.cdouble)         # Memory allocation
Q  = np.zeros((2*N+1, Nlt), dtype=np.cdouble)               # Memory allocation

for i in range(Nlt):
    Q2, W[:,:,i] = np.linalg.eig(A[:,:,i])                  # Eigenvalues Q2 and eigenvectors W for layer i
    Q[:,i] = np.sqrt(Q2)                                    # Calculate Q as square root of eigenvalues

T = Tmatrix(W[:,:,0],Q[:,0],W[:,:,1],Q[:,1])                # Use W and Q of layer 0 and 1 to compute the Tbar0,1 matrix
Sglobal = TtoS(T,Q[:,0],Q[:,1])                             # Convert the Tbar0,1 matrix to S0,1 and call it the global S matrix

for i in range(1,Nlt-1):                                    # For all layers
    T = Tmatrix(W[:,:,i],Q[:,i],W[:,:,i+1],Q[:,i+1])        # Convert to Tbari,i+1 matrix
    S = TtoS(T,Q[:,i],Q[:,i+1])                             # Compute Si,i+1
    Sglobal = Redheffer(Sglobal,S)                          # Use Redheffer to compute S1,i+1

c1_plus = np.ones((2*N+1))
r = np.matmul(Sglobal[2],c1_plus)
t = np.matmul(Sglobal[0],c1_plus)












def EVisual(r,t,x,z):
    kz_n_1   = np.sqrt(k_0*k_0*e0 - Kx*Kx)
    np.sqrt([k_0*k_0*e0-Kx*Kx,1j])
    # kz_n_sub = np.sqrt(k_0*k_0*e0*e1 -Kx*Kx)
    # kz_n_sup = np.sqrt(k_0*k_0*e0*e2 -Kx*Kx)
    # z   = np.arange(start = -H, stop = 2*H, step = H/(10*Nl))
    # x   = np.arange(start = -p/2,stop = p/2, step = p/100)
    E_vis  = np.zeros((2*N+1), dtype=np.cdouble) 
    
    for i in range(2*N+1):
        if i == N:
            z = 1
            x = 1
            E_vis = (np.exp(-1j*kz_n_1*z) + r*np.exp(1j*kz_n_1,z))*np.exp(-1j*Kx*x)
        # else:
        #     E_vis = r*np.exp(1j*kz_n_1*z)*np.exp(-1j*Kx*x)
    
    # for i in range(Nlt):
    #     for j in range(dist.size):
            
    #         if i == 0:
    #             E_vis[i,j] = (t*np.exp(-1j*kz_n_1*z[i]) + r*np.exp(1j*kz_n_1*z[i]))*np.exp(-1j*Kx*dist[j])
                            

    #         if i == Nlt:
    #             E_vis[i,j] = t*np.exp(-1j*kz_n_sub*z[i])*np.exp(-1j*Kx*dist[j]) 
            

    #         else:
    #             E_vis[i,j] = W[mode,mode,i-1]*(np.exp(-k_0*Q[mode,i-1]*z[i-1])*c_plus)*np.exp(-1j*Kx*dist[j]*1)
            
    return E_vis


Efield = EVisual(r,t,1,1)         

def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.show()
    
heatmap2d(np.abs(Efield))