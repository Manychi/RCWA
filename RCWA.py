import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import numpy as np
from functools import reduce # only in Python 3

PI = 3.14159265358979323846

# Inputs #####################################################################
theta_inc = 30.; lambda_0 = 0.02;   # Angle of incidence and wavelength of incident light
p = 1.;                           # Grating period
alpha_l = 30.; alpha_r = 20.;      # Blazing and anti-blazing angle in degrees
d = 0.3;                         # Width of top cut
e0 = 1.; e1 = 2.;  e2 =1.1;        # Dielectric permitivity of surrounding medium and grating

N = 10;                           # Number of harmonics
Nl = 5;                          # Number of layers excluding sub- and superstrate layers
Nlt = Nl+2;                      # Total number of layers including sub- and superstrate layers

accuracy = 100                   # Accuracy to determine the steps in the plots
z_start = -0.5                   # Measure from which the plot starts
z_stop  = 1.5                    # Measure at which height the plot ends





# Geometry calculations ######################################################
alpha_lrad = alpha_l*PI/180; alpha_rrad = alpha_r*PI/180;   # Convert angles to radians
H  = (p-d)/(1/np.tan(alpha_lrad)+1/np.tan(alpha_rrad))      # Height of grating
Hs = H/Nl                                                   # Height of a slice
Sl = -p/2 + Hs/(2*np.tan(alpha_lrad))*(1+2*np.arange(0,Nl)) # x location of the slice left boundary
Sr =  p/2 - Hs/(2*np.tan(alpha_rrad))*(1+2*np.arange(0,Nl)) # x location of the slice right boundary

# Draw Geometry
figure, ax = plt.subplots(1)                                # Make a plot
plt.xlabel("x"); plt.ylabel("z");                           # Plot labels
plt.title("Devision of the grating in rectangles.")         # Plot title
plt.gca().invert_yaxis()                                    # Invert the y-axis of the plot

grating = [[-p/2,H], [p/2,H],   # Make an array consiting of the corner coordinates of the grating
           [p/2-H/np.tan(alpha_rrad),0],[-p/2+H/np.tan(alpha_lrad),0]]
grating.append(grating[0])      # Needed to close the contour of the grating
x, y = zip(*grating)            # Get x and y coordinates separately in a tuple
plt.plot(x,y)                   # Plot the grating shape

for i in range(Nl):             # Draw the rectangular slices
    rect = patches.Rectangle((Sl[i], -Hs*i+H-Hs), -Sl[i] + Sr[i], Hs,edgecolor='r',facecolor="none")
    ax.add_patch(rect)
    
Sl = Sl[::-1]                   # Invert Sl so that top corresponds to layer number 0
Sr = Sr[::-1]                   # Invert Sr so that top corresponds to layer number 0





# Calculation A matrix #######################################################
# Find Kx2 array
k_0 = 2*PI/lambda_0                 # Find k0 from inputs
theta_inc_rad = theta_inc*PI/180    # Convert theta_inc from degrees to radians
kinc_x = k_0*np.cos(theta_inc_rad)  # Find the x component from the incident wave
n = np.arange(-N,N+1)               # Define the range of n
Kx = (kinc_x-2*PI*n/p)/k_0          # Find Kx
Kx2 = Kx*Kx                         # Find the Kx squared matrix diagonal

# Function to solve the integral for all harmonics and put it in an array
def integral(boundaryl,boundaryr):                            # Inputs are the left and right boundary
    result = np.zeros((4*N+1,1), dtype=np.complex64);         # Memory allocation
    for n in range(-2*N,2*N+1):                               # Do the calculation for all harmonics present in the E-matrix
        if n != 0:                                            # n equal to zero gives devision by 0 so must be taken separately
            result[n+2*N] = np.exp(-2j*PI*n*boundaryr/p)*p*1j/(2*PI*n)-np.exp(-2j*PI*n*boundaryl/p)*p*1j/(2*PI*n)      # Calculating the integral
        else:
            result[n+2*N] = boundaryr - boundaryl             # In case n is equal to zero integral is over a constant
    return result                                             # Return the resulting array

# Function to get the Kx2 and Earrays in a matrix form
def ArraysToA(Earray,Kx2):
    A = np.zeros((2*N+1,2*N+1), dtype=np.complex64)           # Memory allocation
    for i in range(4*N+1):                                    # For all harmonics 
        if i < 2*N:
            np.fill_diagonal(A[2*N-i:], -Earray[i]);
        elif i == 2*N:
            np.fill_diagonal(A, Kx2-Earray[i]);
        else:
            np.fill_diagonal(A[:,i-2*N:], -Earray[i]);
    return A                                                  # Return the A matrix for that layer as defined in the slides

# Find the Ai matrix for all layers i
figure, ax = plt.subplots(1)                                  # Make a new plot
plt.xlabel("x"); plt.ylabel("z");                             # Plot labels
plt.title("The normalized and shifted Fourier approximation per slice.") # Plot title

epsilon = np.zeros(accuracy,dtype=np.complex64)               # Memory allocation for the x-dependent permittivity
x = np.zeros(accuracy)                                        # Memory allocation for x
n = np.arange(-2*N,2*N+1).reshape(4*N+1,1)                    # Memory allocation for all the modes used in the sum of the permittivity

A = np.zeros((2*N+1,2*N+1,Nlt), dtype=np.complex64)           # Memory allocation for the total A matrix
A[:,:,0] = ArraysToA(e0*integral(-p/2,p/2),Kx2)               # Determine A in the superstrate
A[:,:,Nlt-1] = ArraysToA(e2*integral(-p/2,p/2),Kx2)           # Determine A in the substrate
   
for i in range(Nl):                                           # For all slices in the geometry          
    Earray = e0*integral(-p/2,Sl[i])+e1*integral(Sl[i],Sr[i])+e0*integral(Sr[i],p/2) # Calculate the complete integral to find epsilon hat in every mode
    A[:,:,i+1] = ArraysToA(Earray,Kx2)                        # Compute the A matrix

    # Compute the inverse Fourier series to plot
    for j in range(accuracy):                                             # For the entire x interval
        x[j] = -p/2 + j*p/accuracy                                        # Calculate the x-coordinate
        epsilon[j] = np.sum(np.multiply(Earray,np.exp(2*PI*1j*n*x[j]/p))) # Calculate the resulting epsilon at that x-coordinate

    epsilon = (epsilon-np.min(epsilon))*Hs/(np.max(epsilon)-np.min(epsilon))+(Nl-i-1)*Hs # Normalize and shift epsilon
    plt.plot(x,np.real(epsilon))                                                         # Plot the approximation of epsilon
    
    # Draw the rectangular slices for the plot
    rect = patches.Rectangle((Sl[i], -Hs*i+H-Hs), -Sl[i] + Sr[i], Hs,edgecolor='r',facecolor="none")
    ax.add_patch(rect)






# Calculate global S matrix and local Tbar matrices ##########################

# Function to find the Tmatrix of interface of layer i and i+1 based on the eigenvalues Q and eigenvectors W
def Tmatrix(Wi,Qi,Wi1,Qi1):
    Tsubi = np.zeros((4*N+2, 4*N+2), dtype=np.complex64)        # Memory allocation
    Tsubi1 = np.zeros((4*N+2, 4*N+2), dtype=np.complex64)       # Memory allocation
    
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
Q2 = np.zeros((2*N+1), dtype=np.complex64)                    # Memory allocation
W  = np.zeros((2*N+1, 2*N+1,Nlt), dtype=np.complex64)         # Memory allocation
Q  = np.zeros((2*N+1, Nlt), dtype=np.complex64)               # Memory allocation
Tbar  = np.zeros((4*N+2, 4*N+2, Nlt-1), dtype=np.complex64)   # Memory allocation

for i in range(Nlt):
    Q2, W[:,:,i] = np.linalg.eig(A[:,:,i])                  # Eigenvalues Q2 and eigenvectors W for layer i
    Q[:,i] = np.sqrt(Q2)                                    # Calculate Q as square root of eigenvalues

Tbari = Tmatrix(W[:,:,0],Q[:,0],W[:,:,1],Q[:,1])                # Use W and Q of layer 0 and 1 to compute the Tbar0,1 matrix
Tbar[:, :, 0] = Tbari
Sglobal = TtoS(Tbari,Q[:,0],Q[:,1])                             # Convert the Tbar0,1 matrix to S0,1 and call it the global S matrix

for i in range(1, Nlt-1):                                    # For all layers
    Tbari = Tmatrix(W[:,:,i],Q[:,i],W[:,:,i+1],Q[:,i+1])        # Convert to Tbari,i+1 matrix
    Tbar[:,:,i] = Tbari
    S = TtoS(Tbari,Q[:,i],Q[:,i+1])                             # Compute Si,i+1
    Sglobal = Redheffer(Sglobal,S)                          # Use Redheffer to compute S0,i+1







# Calculate Cplus and Cminus for every layer #################################


def TbartoTglobal (Tglobal_old, Tbar, Qi, Qi1):
    
    Xi1_matrix = np.zeros((4*N+2, 4*N+2), dtype=np.complex64) 
    Xi_matrix = np.zeros((4*N+2, 4*N+2), dtype=np.complex64) 
    
    Xi1_matrix[0:2*N+1,0:2*N+1] = np.identity(2*N+1)
    Xi1_matrix[2*N+1:4*N+2,2*N+1:4*N+2] = np.diag(np.exp(-k_0*Qi1*Hs))
    
    Xi_matrix[0:2*N+1,0:2*N+1] = np.diag(np.exp(-k_0*Qi*Hs))
    Xi_matrix[2*N+1:4*N+2,2*N+1:4*N+2] = np.identity(2*N+1)
    
    if np.linalg.norm(Tglobal_old) == 0:
        Tglobal_new = reduce(np.matmul, [np.linalg.inv(Xi1_matrix), Tbar, Xi_matrix])
    else:
        Tglobal_new = reduce(np.matmul, [Tglobal_old, np.linalg.inv(Xi1_matrix), Tbar, Xi_matrix])
    
    return Tglobal_new
    
def c_total(Tglobal_new, c1_plus, c1_minus):
       
    T11 = Tglobal_new[0:2*N+1,0:2*N+1]
    T12 = Tglobal_new[2*N+1:4*N+2,0:2*N+1]
    T21 = Tglobal_new[0:2*N+1,2*N+1:4*N+2]
    T22 = Tglobal_new[2*N+1:4*N+2,2*N+1:4*N+2]
    
    cout_plus = np.add(np.matmul(T11, c1_plus), np.matmul(T12, c1_minus))
    cout_minus = np.add(np.matmul(T21, c1_plus), np.matmul(T22, c1_minus))
    
    return cout_plus, cout_minus

c_plus = np.zeros((2*N+1, Nlt), dtype=np.complex64)         # Memory allocation
c_minus = np.zeros((2*N+1, Nlt), dtype=np.complex64)        # Memory allocation
Tglobal  = np.zeros((4*N+2, 4*N+2,Nlt-1), dtype=np.complex64)     # Memory allocation
c_plus[N, 0] = 1                                            # c_plus on first interface has only a 1 in the 0th mode
c_plus[:,Nlt-1], c_minus[:,0] = np.matmul(Sglobal[0],c_plus[:, 0]), np.matmul(Sglobal[2],c_plus[:, 0]) # Find C_M+ and C_1-
r,t=np.matmul(Sglobal[0],c_plus[:, 0]), np.matmul(Sglobal[2],c_plus[:, 0]) #used as check


Tglobal[:,:,0] = TbartoTglobal(Tglobal[:,:,0], Tbar[:,:, 0], Q[:,0], Q[:,1]) #T0,1
c_plus[:,1], c_minus[:,1] = c_total(Tglobal[:,:,0], c_plus[:,0], c_minus[:,0])
 
for i in range(Nl): #Corresponds to the number of interfaces -1
    Tglobal[:,:,i+1] = TbartoTglobal(Tglobal[:,:,i], Tbar[:,:, i+1], Q[:,i+1], Q[:,i+2])
    c_plus[:,i+2], c_minus[:,i+2] = c_total(Tglobal[:,:,i+1], c_plus[:,0], c_minus[:,0])
   
         
    
# def check(check_plus, check_minus):
#     for i in range (np.size(check_plus, 1)-1):
#         if np.add((np.power(np.linalg.norm(check_plus[:, i+1]), 2)),(np.power(np.linalg.norm(check_minus[:, i]), 2))) <= np.add((np.power(np.linalg.norm(check_plus[:, i]), 2)),(np.power(np.linalg.norm(check_minus[:, i+1]), 2))):
#             print("good")
#             #print (np.add((np.power(np.linalg.norm(c_plus[:, i+1]), 2)),(np.power(np.linalg.norm(c_minus[:, i]), 2))))
#             #print (np.add((np.power(np.linalg.norm(c_plus[:, i]), 2)),(np.power(np.linalg.norm(c_minus[:, i+1]), 2))))
#         else:
#             print ("start crying")   
#             print (np.add((np.power(np.linalg.norm(c_plus[:, i+1]), 2)),(np.power(np.linalg.norm(c_minus[:, i]), 2))))
#             print (np.add((np.power(np.linalg.norm(c_plus[:, i]), 2)),(np.power(np.linalg.norm(c_minus[:, i+1]), 2))))
#     return

# check (c_plus, c_minus)

# r = np.matmul(Sglobal[2],c_plus[:, 0])
# t = np.matmul(Sglobal[0],c_plus[:, 0])


# #Function to find layer and heights of z-boundaries
# def FindLayer(Height):
#     for i in range(Nlt):            #Range for each layer
#         if Height<= H/Nl*i:         #Check whether given height is below boundary
#             z_begin = H/Nl*(i-1)    #Calcs the lower boundary z
#             z_next  = H/Nl*i        #Calcs the upper boundary z
#             return i-1,z_begin,z_next
        
        
# #Function to calculate the E-field given a certain position in z and x with each mode
# def EVisual(r,t,c_plus,c_min):
#     z   = np.arange(start = z_start, stop = z_stop, step = (z_stop-z_start)/(accuracy))   #Defines the z axis
#     x   = np.arange(start = -p/2,stop = p/2, step = p/accuracy)           #Defines the x axis
#     E_vis  = np.zeros((z.size,x.size), dtype=np.complex64)             #Memory allocation for E_vis
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
#             if z[j] >= 0 and z[j] <= H:             #check wheter we are in grating
#                  layer,z0,z1 = FindLayer(z[j])      #Finds the layer, and heights of boundaries
#                  for i in range(2*N+1):             #Makes loop for all modes in 1 direction
#                     for k in range(2*N+1):          #Makes loop for all modes in other direction
                        
#                         # kz_n_grat = np.sqrt([(k_0*k_0*e0*e1 - Kx[i]*Kx[i]),(1-1j)])  
#                         E_vis[j,l] = (W[i,k,layer]*np.exp(-1*k_0*Q[k,layer]*(z[j]-z0))*c_plus[i,layer] 
#                         + c_min[i,layer]*np.exp(k_0*Q[k,layer]*(z[j]-z1)))*np.exp(-1j*Kx[i]*x[l]) #Calculates E field each mode
                        
#                         E_vis[j,l] += E_vis[j,l]        #Addition to get E_vis for each mode on a position 
                        
#             else:                       #Substrate is the only position left
#                 for i in range(2*N+1):  #Makes loop for all modes
#                     kz_n_sub= np.sqrt([(k_0*k_0*e0*e2 - Kx[i]*Kx[i]),(1-1j)])                   #Calculates the k_{z,n} for each mode
#                     E_vis[j,l] = t[i]*np.exp(-1j*kz_n_sub[0]*(z[j]-H))*np.exp(-1j*Kx[i]*x[l])   #Calculates E field each mode
#                     E_vis[j,l] += E_vis[j,l]            #Addition to get E_vis for each mode on a position
                    
                
#     return E_vis

# Efield = EVisual(r,t,c_plus,c_minus)         
# z   = np.arange(start = z_start, stop = z_stop, step = (z_stop-z_start)/(accuracy))   #Defines the z axis
# x   = np.arange(start = -p/2,stop = p/2, step = p/100)           #Defines the x axis
    
# def heatmap2d(arr: np.ndarray):
#     plt.imshow(arr, cmap='inferno')
#     plt.colorbar()
#     plt.show()
    
# def GetTickValues(array,nrlabels):
#     j = 0
#     Ticks = np.zeros(nrlabels+1,dtype = float)
#     step = round(array.size/nrlabels)
#     for i in range(nrlabels+1):
#             if i == nrlabels:
#                 Ticks[j] = array[array.size-1]
#                 return Ticks
#             Ticks[j] = array[step*i]
#             j +=1
    
# def heatmapEfield(Efield):
#     figure, ax = plt.subplots(1)
    
#     for i in range(Nl):             # Draw the rectangular slices
#         rect = patches.Rectangle((Sl[Nl-1-i]*x.size+x.size/2, (i)*z.size/(Nlt)+(z_stop-z_start)/(2*H*Nlt)*z.size), 
#                                  (-Sl[Nl-1-i] + Sr[Nl-1-i])*x.size, (z.size)/Nlt ,
#                                  edgecolor='b',facecolor="none")
#         print((i)*z.size/Nlt ,np.around(((z.size)/Nlt)/Nl,2))
#         ax.add_patch(rect)
#     # plt.plot( c='g',zorder=10)
#     hm = ax.matshow(Efield,cmap ='inferno')
#     # ax.imshow(Efield, extent=[-p/2,p/2,H,0])
#     nx = x.shape[0]
#     no_xlabels = 5 # how many labels visible on the x axis
#     step_x = int(nx/(no_xlabels-1)) # step between consecutive labels
#     x_positions = np.arange(0,nx,step_x-1/no_xlabels) # pixel count at label position
#     x_labels = np.around(x[::step_x],2) # labels you want to see
#     x_labels = np.append(x_labels,p/2)
#     ny = z.shape[0]
#     no_ylabels = 5 # how many labels visible on the y axis
#     step_y = int(ny/(no_ylabels-1)) 
#     y_positions = np.arange(0,ny,step_y-1/no_ylabels)
#     y_labels =z[::step_y] # labels you want to see
#     y_labels =  np.around(np.append(y_labels,z_stop),4)*1000
    
#     plt.xticks(x_positions, x_labels)
#     plt.yticks(y_positions, y_labels)
#     plt.xlabel('x  [nm]', fontweight ='bold') 
#     plt.ylabel('z, [nm]', fontweight ='bold')
#     ax.get_xaxis().tick_bottom()
#     figure.colorbar(hm)
#     plt.show()

# heatmap2d(np.abs(Efield))
# heatmapEfield(np.abs(Efield))