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

N = 6;                         # Number of harmonics
Nl = 15;                        # Number of layers excluding sub- and superstrate layers
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
    Sglobal_new11 = reduce(np.matmul, [S[0], np.linalg.inv(np.add(np.identity(2*N+1),-np.matmul(Sglobal_old[1], S[2]))) ,Sglobal_old[0]]) 
    Sglobal_new12 = np.add(S[1], reduce(np.matmul,[S[0], Sglobal_old[1], np.linalg.inv(np.add(np.identity(2*N+1),-np.matmul(S[2], Sglobal_old[1]))) ,S[3]])) 
    Sglobal_new21 = np.add(Sglobal_old[2], reduce(np.matmul,[Sglobal_old[3], S[2], np.linalg.inv(np.add(np.identity(2*N+1),-np.matmul(Sglobal_old[1], S[2]))) ,Sglobal_old[0]])) 
    Sglobal_new22 = reduce(np.matmul, [Sglobal_old[3], np.linalg.inv(np.add(np.identity(2*N+1),-np.matmul(S[2], Sglobal_old[1]))) ,S[3]])  
    return np.array([Sglobal_new11, Sglobal_new12, Sglobal_new21, Sglobal_new22])                  # Return all the quadrants of the computed global S-matrix

def c_variables(Sglobal1, c1_plus, cm_minus):
    c =np.array([np.add(np.matmul(Sglobal[0], c1_plus), np.matmul(Sglobal[1], cm_minus)), np.add(np.matmul(Sglobal[2], c1_plus), np.matmul(Sglobal[3], cm_minus))])
    c1_plus = c[0]
    cm_minus = c[1]
    return c1_plus, cm_minus


def c_total(Tglobal, c1_plus, c1_minus, Qi, Qi1):
       
    Xi1_matrix = np.zeros((4*N+2, 4*N+2), dtype=np.cdouble) 
    Xi_matrix = np.zeros((4*N+2, 4*N+2), dtype=np.cdouble) 
    
    Xi1_matrix[0:2*N+1,0:2*N+1] = np.identity(2*N+1)
    Xi1_matrix[2*N+1:4*N+2,2*N+1:4*N+2] = np.linalg.inv(np.diag(np.exp(-k_0*Qi1*Hs)))
    
    Xi_matrix[0:2*N+1,0:2*N+1] = np.diag(np.exp(-k_0*Qi*Hs))
    Xi_matrix[2*N+1:4*N+2,2*N+1:4*N+2] = np.identity(2*N+1)
    
    Tglobal = reduce(np.matmul, [Xi1_matrix, Tglobal, Xi_matrix])
    
    T11 = Tglobal[0:2*N+1,0:2*N+1]
    T12 = Tglobal[2*N+1:4*N+2,0:2*N+1]
    T21 = Tglobal[0:2*N+1,2*N+1:4*N+2]
    T22 = Tglobal[2*N+1:4*N+2,2*N+1:4*N+2]
    
    cout_plus = np.add(np.matmul(T11, c1_plus), np.matmul(T12, c1_minus))
    cout_minus = np.add(np.matmul(T21, c1_plus), np.matmul(T22, c1_minus))
    
    return cout_plus, cout_minus

def check(check_plus, check_minus):
    for i in range (np.size(check_plus, 1)-1):
        if np.add((np.power(np.linalg.norm(check_plus[:, i+1]), 2)),(np.power(np.linalg.norm(check_minus[:, i]), 2))) <= np.add((np.power(np.linalg.norm(check_plus[:, i]), 2)),(np.power(np.linalg.norm(check_minus[:, i+1]), 2))):
            print("good")
            #print (np.add((np.power(np.linalg.norm(c_plus[:, i+1]), 2)),(np.power(np.linalg.norm(c_minus[:, i]), 2))))
            #print (np.add((np.power(np.linalg.norm(c_plus[:, i]), 2)),(np.power(np.linalg.norm(c_minus[:, i+1]), 2))))
        else:
            print ("start crying")   
            print (np.add((np.power(np.linalg.norm(c_plus[:, i+1]), 2)),(np.power(np.linalg.norm(c_minus[:, i]), 2))))
            print (np.add((np.power(np.linalg.norm(c_plus[:, i]), 2)),(np.power(np.linalg.norm(c_minus[:, i+1]), 2))))
    return


# Calculate the eigenvalues and eigenvectors for every layer
Q2 = np.zeros((2*N+1), dtype=np.cdouble)                    # Memory allocation
W  = np.zeros((2*N+1, 2*N+1,Nlt), dtype=np.cdouble)         # Memory allocation
Q  = np.zeros((2*N+1, Nlt), dtype=np.cdouble)               # Memory allocation
Tbar  = np.zeros((4*N+2, 4*N+2, Nlt-1), dtype=np.cdouble)  # Memory allocation

for i in range(Nlt):
    Q2, W[:,:,i] = np.linalg.eig(A[:,:,i])                  # Eigenvalues Q2 and eigenvectors W for layer i
    Q[:,i] = np.sqrt(Q2)                                    # Calculate Q as square root of eigenvalues

T = Tmatrix(W[:,:,0],Q[:,0],W[:,:,1],Q[:,1])                # Use W and Q of layer 0 and 1 to compute the Tbar0,1 matrix
Tbar[:, :, 0] = T
Sglobal = TtoS(T,Q[:,0],Q[:,1])                             # Convert the Tbar0,1 matrix to S0,1 and call it the global S matrix

for i in range(1, Nlt-1):                                    # For all layers
    T = Tmatrix(W[:,:,i],Q[:,i],W[:,:,i+1],Q[:,i+1])        # Convert to Tbari,i+1 matrix
    Tbar[:,:,i] = T
    S = TtoS(T,Q[:,i],Q[:,i+1])                             # Compute Si,i+1
    Sglobal = Redheffer(Sglobal,S)                          # Use Redheffer to compute S1,i+1

c_plus = np.zeros((2*N+1, Nlt), dtype=np.cdouble)         # Memory allocation
c_minus = np.zeros((2*N+1, Nlt), dtype=np.cdouble)        # Memory allocation
c_plus[:, 0] = np.ones((2*N+1))                             # Input of C plus matrix (ones)

c_plus[:,Nlt-1], c_minus[:,0] = c_variables(Sglobal, c_plus[:,0], c_minus[:, Nlt-1])


for i in range(1, Nlt-1):
    c_plus[:,i], c_minus[:,i] = c_total(Tbar[:,:, i-1], c_plus[:,i-1], c_minus[:,i-1], Q[:,i-1], Q[:,i])

check (c_plus, c_minus)

r = np.matmul(Sglobal[2],c_plus[:, 0])
t = np.matmul(Sglobal[0],c_plus[:, 0])

#Function to find layer and heights of z-boundaries
def FindLayer(Height):
    for i in range(Nlt):            #Range for each layer
        if Height<= H/Nl*i:         #Check whether given height is below boundary
            z_begin = H/Nl*(i-1)    #Calcs the lower boundary z
            z_next  = H/Nl*i        #Calcs the upper boundary z
            return i,z_begin,z_next
        
        
# Function to calculate the E-field given a certain position in z and x with each mode
# def EVisual(r,t,c_plus,c_min):
#     z   = np.arange(start = -0.5*H, stop = 2*H, step = H/(100*Nl))   #Defines the z axis
#     x   = np.arange(start = -p/2,stop = p/2, step = p/100)       #Defines the x axis
#     E_vis  = np.zeros((z.size,x.size), dtype=np.cdouble)        #Memory allocation for E_vis
#     for l in range(x.size):             #Makes the range for all x
        
#         for j in range(z.size):         #Makes the range for all z
#             if z[j] <= 0:               #Checks whether we are in superstrate
            
#                 for i in range(2*N+1):  #Makes loop for all modes
#                     kz_n_sup= np.sqrt([(k_0*k_0*e0 - Kx[i]*Kx[i]),(1-1j)]) #Calculates the k_{z,n} for each mode
            
#                     if i == N:                                             #Calculates E field of mode 0
#                         E_vis[j,l] = (np.exp(-1j*kz_n_sup[0]*z[j]) + r[i]*np.exp(1j*kz_n_sup[0]*z[j]))*np.exp(-1j*Kx[i]*x[l])
#                     else:                                                  #Calculates E field other modes
#                         E_vis[j,l] = r[i]*np.exp(1j*kz_n_sup[0]*z[j])*np.exp(-1j*Kx[i]*x[l])
#                     E_vis[j,l] += E_vis[j,l]                               #Addition to get E_vis for each mode on a position
                    
#             if z[j]>=H:                 #Checks whether we are in substrate
#                 for i in range(2*N+1):  #Makes loop for all modes
#                     kz_n_sub= np.sqrt([(k_0*k_0*e0*e2 - Kx[i]*Kx[i]),(1-1j)])               #Calculates the k_{z,n} for each mode
#                     E_vis[j,l] = t[i]*np.exp(-1j*kz_n_sub[0]*z[j])*np.exp(-1j*Kx[i]*x[l])   #Calculates E field each mode
#                     E_vis[j,l] += E_vis[j,l]            #Addition to get E_vis for each mode on a position
                    
#             else:                           #Now we are in the grating as we are not in sub or sup
#                 for i in range(2*N+1):      #Makes loop for all modes in 1 direction
#                     for k in range(2*N+1):  #Makes loop for all modes in other direction
                        
#                         # kz_n_grat = np.sqrt([(k_0*k_0*e0*e1 - Kx[i]*Kx[i]),(1-1j)])
#                         layer,z0,z1 = FindLayer(z[j])   #Finds the layer, and heights of boundaries 
#                         E_vis[j,l] = (W[i,k,layer]*np.exp(-1*k_0*Q[k,layer]*-(z[j]-z0))*c_plus[layer] 
#                         + c_min[layer]*np.exp(-k_0*Q[k,layer]*(z[j]-z1)))*np.exp(-1j*Kx[i]*x[layer]) #Calculates E field each mode
                        
#                         E_vis[j,l] += E_vis[j,l]        #Addition to get E_vis for each mode on a position
#     return E_vis


# Efield = EVisual(r,t,c_plus,c_minus)         

# def heatmap2d(arr: np.ndarray):
#     plt.imshow(arr, cmap='viridis')
#     plt.colorbar()
#     plt.show()
    
# heatmap2d(np.abs(Efield))