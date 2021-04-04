import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import matplotlib.image as mpimg
import numpy as np
from functools import reduce # only in Python 3

PI = np.pi

# '''INPUTS'''
# theta_inc = 10; lambda_0 = 0.4;  # Angle of incidence and wavelength of incident light
# p = 1;                           # Grating period
# alpha_l = 20.; alpha_r = 40.;    # Blazing and anti-blazing angle in degrees
# d = 0.5;                         # Width of top cut
# e0 = 1; e1 = 2;  e2 =1;          # Dielectric permitivity of surrounding medium and grating

# N = 10;                           # Number of harmonics
# Nl = 5;                          # Number of layers excluding sub- and superstrate layers
# Nlt = Nl+2;                      # Total number of layers including sub- and superstrate layers

# accuracy = 100                   # Accuracy to determine the steps in the plots



# Function to solve the integral for all harmonics and put it in an array
def integral(boundaryl,boundaryr,N,p):                            # Inputs are the left and right boundary
    result = np.zeros((4*N+1,1), dtype=np.complex64);         # Memory allocation
    for n in range(-2*N,2*N+1):                               # Do the calculation for all harmonics present in the E-matrix
        if n != 0:                                            # n equal to zero gives devision by 0 so must be taken separately
            result[n+2*N] = np.exp(-2j*PI*n*boundaryr/p)*p*1j/(2*PI*n)-np.exp(-2j*PI*n*boundaryl/p)*p*1j/(2*PI*n)      # Calculating the integral
        else:
            result[n+2*N] = boundaryr - boundaryl             # In case n is equal to zero integral is over a constant
    return result                                             # Return the resulting array

# Function to get the Kx2 and Earrays in a matrix form
def ArraysToA(Earray,Kx2,N):
    A = np.zeros((2*N+1,2*N+1), dtype=np.complex64)           # Memory allocation
    for i in range(4*N+1):                                    # For all harmonics
        if i < 2*N:
            np.fill_diagonal(A[2*N-i:], -Earray[i]);
        elif i == 2*N:
            np.fill_diagonal(A, Kx2-Earray[i]);
        else:
            np.fill_diagonal(A[:,i-2*N:], -Earray[i]);
    return A                                                  # Return the A matrix for that layer as defined in the slides


'''
Two functions:
    One, to calculate the full T_Matrix from Wi+1, Wi, the Q's and X's
    The second to calculate the S matrix from the T_matrix
'''
def T_Matrix_WQX(W1, Q1, W2, Q2, X1, X2,N):
    T_sub1 = np.zeros((4*N+2, 4*N+2), dtype=np.complex64)        # Memory allocation
    T_sub2 = np.zeros((4*N+2, 4*N+2), dtype=np.complex64)       # Memory allocation
    
    T_sub1[0:2*N+1,0:2*N+1] = np.matmul(W1,X1)                              #Fill T11
    T_sub1[0:2*N+1,2*N+1:4*N+2] = W1                                        #Fill T12
    T_sub1[2*N+1:4*N+2,0:2*N+1] = np.matmul(np.matmul(W1,np.diag(Q1)),X1)   #Fill T21
    T_sub1[2*N+1:4*N+2,2*N+1:4*N+2] = -1*np.matmul(W1,np.diag(Q1))          #Fill T22
    #repeat for next layer
    T_sub2[0:2*N+1,0:2*N+1] = W2
    T_sub2[0:2*N+1,2*N+1:4*N+2] = np.matmul(W2,X2)
    T_sub2[2*N+1:4*N+2,0:2*N+1] = np.matmul(W2,np.diag(Q2))    
    T_sub2[2*N+1:4*N+2,2*N+1:4*N+2] = -1*np.matmul(np.matmul(W2,np.diag(Q2)),X2)
    
    return np.matmul(np.linalg.inv(T_sub2),T_sub1), T_sub1, T_sub2

def T_Matrix_WQ(W1, Q1, W2, Q2,N):
    T_sub1 = np.zeros((4*N+2, 4*N+2), dtype=np.complex64)        # Memory allocation
    T_sub2 = np.zeros((4*N+2, 4*N+2), dtype=np.complex64)       # Memory allocation
    
    T_sub1[0:2*N+1,0:2*N+1] = W1                              #Fill T11
    T_sub1[0:2*N+1,2*N+1:4*N+2] = W1                                        #Fill T12
    T_sub1[2*N+1:4*N+2,0:2*N+1] = np.matmul(W1,np.diag(Q1))   #Fill T21
    T_sub1[2*N+1:4*N+2,2*N+1:4*N+2] = -1*np.matmul(W1,np.diag(Q1))          #Fill T22
    #repeat for next layer
    T_sub2[0:2*N+1,0:2*N+1] = W2
    T_sub2[0:2*N+1,2*N+1:4*N+2] = W2
    T_sub2[2*N+1:4*N+2,0:2*N+1] = np.matmul(W2,np.diag(Q2))    
    T_sub2[2*N+1:4*N+2,2*N+1:4*N+2] = -1*np.matmul(W2,np.diag(Q2))
    
    return np.matmul(np.linalg.inv(T_sub2),T_sub1), T_sub1, T_sub2
    
    
def T_Mat_to_S_Mat(T_Matrix,N):
    S_Mat = np.zeros((4*N+2, 4*N+2), dtype=np.complex64)
    #Load all points, T11, T12 etc.
    T11 = T_Matrix[0:2*N+1,0:2*N+1]
    T21 = T_Matrix[2*N+1:4*N+2,0:2*N+1]
    T12 = T_Matrix[0:2*N+1,2*N+1:4*N+2]
    T22 = T_Matrix[2*N+1:4*N+2,2*N+1:4*N+2]
    #Calculate all S_matrix solutions S11, S12, S21, S22
    S_Mat[0:2*N+1,0:2*N+1] = T11 - reduce(np.matmul,[T12,np.linalg.inv(T22),T21])
    S_Mat[0:2*N+1,2*N+1:4*N+2] = np.matmul(T12,np.linalg.inv(T22))
    S_Mat[2*N+1:4*N+2,0:2*N+1] = -np.matmul(np.linalg.inv(T22),T21)
    S_Mat[2*N+1:4*N+2,2*N+1:4*N+2] = np.linalg.inv(T22)
    return S_Mat

def S_Mat_Bar_to_S(S_Matrix,i,Xi,N):
    X1 = Xi[:,:,i]
    X2 = Xi[:,:,i+1]
    Filler = np.zeros((2*N+1, 2*N+1), dtype=np.complex64)
    X_Multiply = Sections_to_Full(np.array([X1, Filler, Filler, X2]))
    S_Mat = np.matmul(S_Matrix,X_Multiply)    
    return S_Mat

def S_Mat_Direct(T_Matrix,i,Xi,N):
    T_Parts = Full_to_Sections(T_Matrix)
    S_Mat = np.zeros((4*N+2, 4*N+2), dtype=np.complex64)
    X1 = Xi[:,:,i]
    X2 = Xi[:,:,i+1]
    S_Mat[0:2*N+1,0:2*N+1] =np.matmul((T_Parts[0] - reduce(np.matmul,[T_Parts[1],np.linalg.inv(T_Parts[3]),T_Parts[2]])),X1)
    S_Mat[0:2*N+1,2*N+1:4*N+2] = np.matmul(np.matmul(T_Parts[1],np.linalg.inv(T_Parts[3])),X2)
    S_Mat[2*N+1:4*N+2,0:2*N+1] = np.matmul(-np.matmul(np.linalg.inv(T_Parts[3]),T_Parts[2]),X1)
    S_Mat[2*N+1:4*N+2,2*N+1:4*N+2] = np.matmul(np.linalg.inv(T_Parts[3]),X2)
    return S_Mat
    
'''
Down here are a few functions to go from any array of T or S to either a 
full array or go back to T11, T12 etc.

''' 
def Sections_to_Full(Array):
    layers = Array.shape[0]
    lengths = Array.shape[1]
    
    Full_Array = np.zeros((2*lengths, 2*lengths), dtype=np.complex64)
    for i in range(layers):
        if i <=1 :
            Full_Array[0:lengths,0+i*lengths:lengths*(i+1)] = Array[i]
        else:
            Full_Array[lengths:lengths*2, 0+np.mod(i,2)*lengths:lengths*(np.mod(i,2)+1) ] = Array[i]
    
    return Full_Array

def Full_to_Sections(Array,N):
    layers = 4
    lengths = 2*N+1
    
    Section_Array = np.zeros((4,lengths, lengths),dtype=np.complex64)
    
    for i in range(layers):
        if i <= 1:
            Section_Array[i] = Array[0:lengths,0+i*lengths:lengths*(i+1)]
        else:
            Section_Array[i] = Array[lengths:lengths*2,0+np.mod(i,2)*lengths:lengths*(np.mod(i,2)+1)]  
    return Section_Array


# Function to calculate the Redheffer star product from all quadrants of S1,i and Si,i+1
# The indeces between brackets of the S-matrices correspond to S11, S12, S21, S22 respectively
def Redheffer(Sglobal_old,S,N):  #Sglobal is the total S matrix, S is the next layer
    Sglobal_new11 = reduce(np.matmul, [S[0], np.linalg.inv(np.identity(2*N+1)-np.matmul(Sglobal_old[1], S[2])) ,Sglobal_old[0]]) 
    Sglobal_new12 = np.add(S[1], reduce(np.matmul,[S[0], Sglobal_old[1], np.linalg.inv(np.identity(2*N+1)-np.matmul(S[2], Sglobal_old[1])) ,S[3]])) 
    Sglobal_new21 = np.add(Sglobal_old[2], reduce(np.matmul,[Sglobal_old[3], S[2], np.linalg.inv(np.identity(2*N+1)-np.matmul(Sglobal_old[1], S[2])) ,Sglobal_old[0]])) 
    Sglobal_new22 = reduce(np.matmul, [Sglobal_old[3], np.linalg.inv(np.identity(2*N+1)-np.matmul(S[2], Sglobal_old[1])) ,S[3]])  
    return np.array([Sglobal_new11, Sglobal_new12, Sglobal_new21, Sglobal_new22])                  # Return all the quadrants of the computed global S-matrix


def check(check_plus, check_minus,Nlt):
    check = np.zeros(Nlt)
    for i in range (np.size(check_plus, 1)-1):
        check[i] = np.power(np.linalg.norm(check_minus[:, i]), 2)-np.power(np.linalg.norm(check_plus[:, i]), 2)
    plt.plot(check)
    return

#Function to find layer and heights of z-boundaries
def FindLayer(Height,H,Nl):
    Nlt = Nl+2
    for i in range(Nlt):            #Range for each layer
        if Height<= H/Nl*i:         #Check whether given height is below boundary
            z_begin = H/Nl*(i-1)    #Calcs the lower boundary z
            z_next  = H/Nl*i        #Calcs the upper boundary z
            return i-1,z_begin,z_next
        
        
#Function to calculate the E-field given a certain position in z and x with each mode
def EVisual(r,t,c_plus,c_min,z_start,z_stop,k_0,Kx,H,W,Q,accuracy,p,N,e0,e2,Nl):
    z   = np.arange(start = z_start, stop = z_stop, step = (z_stop-z_start)/(accuracy))   #Defines the z axis
    x   = np.arange(start = -p/2,stop = p/2, step = p/accuracy)           #Defines the x axis
    E_vis  = np.zeros((z.size,x.size), dtype=np.complex64)             #Memory allocation for E_vis
    for l in range(x.size):             #Makes the range for all x
        
        for j in range(z.size):         #Makes the range for all z
            if z[j] <= 0:               #Checks whether we are in superstrate
            
                for i in range(2*N+1):  #Makes loop for all modes
                    kz_n_sup= np.sqrt([(k_0*k_0*e0 - Kx[i]*Kx[i]*k_0*k_0),(1-1j)]) #Calculates the k_{z,n} for each mode
            
                    if i == N:                                             #Calculates E field of mode 0
                        E_vis[j,l] = (np.exp(-1j*kz_n_sup[0]*z[j]) + r[i]*np.exp(1j*kz_n_sup[0]*z[j]))*np.exp(-1j*Kx[i]*k_0*x[l])
                    else:                                                  #Calculates E field other modes
                        E_vis[j,l] = r[i]*np.exp(1j*kz_n_sup[0]*z[j])*np.exp(-1j*Kx[i]*k_0*x[l])
                    E_vis[j,l] += E_vis[j,l]                               #Addition to get E_vis for each mode on a position
            if z[j] >= 0 and z[j] <= H:             #check wheter we are in grating
                  layer,z0,z1 = FindLayer(z[j],H,Nl)      #Finds the layer, and heights of boundaries
                  for i in range(2*N+1):             #Makes loop for all modes in 1 direction
                    for k in range(2*N+1):          #Makes loop for all modes in other direction
                        
                        # kz_n_grat = np.sqrt([(k_0*k_0*e0*e1 - Kx[i]*Kx[i]),(1-1j)])  
                        E_vis[j,l] = (W[i,k,layer]*np.exp(-1*k_0*Q[k,layer]*(z[j]-z0))*c_plus[i,layer] +\
                                      c_min[i,layer]*np.exp(k_0*Q[k,layer]*(z[j]-z1)))*np.exp(-1j*Kx[i]*k_0*x[l]) #Calculates E field each mode
                        
                        E_vis[j,l] += E_vis[j,l]        #Addition to get E_vis for each mode on a position 
                        
            else:                       #Substrate is the only position left
                for i in range(2*N+1):  #Makes loop for all modes
                    kz_n_sub= np.sqrt([(k_0*k_0*e0*e2 - Kx[i]*Kx[i]),(1-1j)])                   #Calculates the k_{z,n} for each mode
                    E_vis[j,l] = t[i]*np.exp(-1j*kz_n_sub[0]*(z[j]-H))*np.exp(-1j*Kx[i]*k_0*x[l])   #Calculates E field each mode
                    E_vis[j,l] += E_vis[j,l]            #Addition to get E_vis for each mode on a position
                    
                
    return E_vis
def heatmapEfield(Efield,x,H,z_start,z_stop,z,Sl,Sr,Nl,p):
    figure, ax = plt.subplots(1)
    nx = x.shape[0]
    no_xlabels = 5 
    for i in range(Nl):             # Draw the rectangular slices
        pos_y = i*H/(z_stop-z_start)*z.size/Nl+int(nx/(no_xlabels-1))
        pos_y_diff = H/(z_stop-z_start)*(z.size)/Nl
        rect = patches.Rectangle((Sl[i]*x.size+x.size/2, pos_y), 
                                 ( -Sl[i] + Sr[i])*x.size, pos_y_diff ,
                                 edgecolor='r',facecolor="none")
        print(pos_y,np.around(pos_y_diff,2))
        ax.add_patch(rect)
    hm = ax.matshow(Efield,cmap ='inferno')
    # ax.imshow(Efield, extent=[-p/2,p/2,H,0])
    nx = x.shape[0]
    no_xlabels = 5                                      # how many labels visible on the x axis
    step_x = int(nx/(no_xlabels-1))                     # step between consecutive labels
    x_positions = np.arange(0,nx,step_x-1/no_xlabels)   # pixel count at label position
    x_labels = np.around(x[::step_x],2) # labels you want to see
    x_labels = np.append(x_labels,p/2)
    ny = z.shape[0]
    no_ylabels = 5 # how many labels visible on the y axis
    step_y = int(ny/(no_ylabels-1)) 
    y_positions = np.arange(0,ny,step_y-1/no_ylabels)
    y_labels =z[::step_y] # labels you want to see
    y_labels =  np.around(np.append(y_labels,z_stop),4)
    
    plt.xticks(x_positions, x_labels)
    plt.yticks(y_positions, y_labels)
    plt.xlabel('x  [nm]', fontweight ='bold') 
    plt.ylabel('z, [nm]', fontweight ='bold')
    ax.get_xaxis().tick_bottom()
    figure.colorbar(hm)
    plt.show()

def solver(theta_inc, lambda_0, p, alpha_l, alpha_r, d, e0,e1,e2,N,Nl,accuracy):
    Nlt = Nl+2
    '''GEOMETRY CALCULATIONS'''
    
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
    
    z_start = -0.5*H                # Measure from which the plot starts
    z_stop  = 1.5*H                 # Measure at which height the plot ends
    
    
    ''' Calculation A matrix '''
    
    # Find Kx2 array
    k_0 = 2*PI/lambda_0                 # Find k0 from inputs
    theta_inc_rad = theta_inc*PI/180    # Convert theta_inc from degrees to radians
    kinc_x = k_0*np.cos(theta_inc_rad)  # Find the x component from the incident wave
    n = np.arange(-N,N+1)               # Define the range of n
    Kx = (kinc_x-2*PI*n/p)/k_0          # Find Kx
    Kx2 = Kx*Kx                         # Find the Kx squared matrix diagonal
    
    # Find the Ai matrix for all layers i
    # figure, ax = plt.subplots(1)                                  # Make a new plot
    # plt.xlabel("x"); plt.ylabel("z");                             # Plot labels
    # plt.title("The normalized and shifted Fourier approximation per slice.") # Plot title
    # plt.gca().invert_yaxis()                                      # Invert the y-axis of the plot
    
    epsilon = np.zeros(accuracy,dtype=np.complex64)               # Memory allocation for the x-dependent permittivity
    x = np.zeros(accuracy)                                        # Memory allocation for x
    n = np.arange(-2*N,2*N+1).reshape(4*N+1,1)                    # Memory allocation for all the modes used in the sum of the permittivity
    
    A = np.zeros((2*N+1,2*N+1,Nlt), dtype=np.complex64)           # Memory allocation for the total A matrix
    A[:,:,0] = ArraysToA(integral(-p/2,p/2,N,p),Kx2,N) 
    A[:,:,Nlt-1] = ArraysToA(integral(-p/2,p/2,N,p),Kx2,N) 
    for i in range(Nl):                                           # For all slices in the geometry          
        Earray = e0*integral(-p/2,Sl[i],N,p)+e1*integral(Sl[i],Sr[i],N,p)+e0*integral(Sr[i],p/2,N,p) # Calculate the complete integral to find epsilon hat in every mode
        A[:,:,i+1] = ArraysToA(Earray,Kx2,N)                        # Compute the A matrix
    
        # Compute the inverse Fourier series to plot
        for j in range(accuracy):                                             # For the entire x interval
            x[j] = -p/2 + j*p/accuracy                                        # Calculate the x-coordinate
            epsilon[j] = np.sum(np.multiply(Earray,np.exp(2*PI*1j*n*x[j]/p))) # Calculate the resulting epsilon at that x-coordinate
        
        epsilon = -(epsilon-np.min(epsilon))*Hs/(np.max(epsilon)-np.min(epsilon))+(i+1)*Hs   # Normalize and shift epsilon
        # plt.plot(x,np.real(epsilon))                                                         # Plot the approximation of epsilon
        
        # Draw the rectangular slices for the plot
        rect = patches.Rectangle((Sl[-i-1], -Hs*i+H-Hs), -Sl[-i-1] + Sr[-i-1], Hs,edgecolor='r',facecolor="none")
        ax.add_patch(rect)
    
        
    '''
    Memory allocations for functions to come
    '''
    T_Matrix    = np.zeros((4*N+2, 4*N+2, Nlt-1), dtype=np.complex64)
    # T_Sub1      = np.zeros((4*N+2, 4*N+2, Nlt-1), dtype=np.complex64)
    # T_Sub2      = np.zeros((4*N+2, 4*N+2, Nlt-1), dtype=np.complex64)
    
    T_Bar_Mat   =  np.zeros((4*N+2, 4*N+2, Nlt-1), dtype=np.complex64)
    T_BSub1     =  np.zeros((4*N+2, 4*N+2, Nlt-1), dtype=np.complex64)
    T_BSub2     =  np.zeros((4*N+2, 4*N+2, Nlt-1), dtype=np.complex64)
    
    # S_Matrix    = np.zeros((4*N+2, 4*N+2, Nlt-1), dtype=np.complex64)
    S_Bar_Matrix= np.zeros((4*N+2, 4*N+2, Nlt-1), dtype=np.complex64)
    # test_full   = np.zeros((4*N+2, 4*N+2), dtype=np.complex64)
    S_Mat_Stable= np.zeros((4*N+2, 4*N+2, Nlt-1), dtype=np.complex64)
    S_Redheffd  = np.zeros((4*N+2, 4*N+2), dtype = np.complex64)
    S_Redheff_test = np.zeros((4*N+2, 4*N+2, Nlt-1), dtype=np.complex64)
    # S_RedhefS_test = np.zeros((4*N+2, 4*N+2, Nlt-1), dtype=np.complex64)
    # S_Mat_Dir= np.zeros((4*N+2, 4*N+2, Nlt-1), dtype=np.complex64)
    
    '''
    For loop to Calculate all T matrices, and all S-Matrices. 
    For every iteration of the S matrix the Redheffer is calculated 
    When this Redheffer is calcualted each layer is stored in S_Redheff_test for debugging purposes
    The if statements are set in place to see if some continuities should hold
    
    '''
    
    # Calculate the eigenvalues and eigenvectors for every layer
    Q2 = np.zeros((2*N+1), dtype=np.complex64)                    # Memory allocation for the eigenvalues
    Q2_new = np.zeros((2*N+1), dtype=np.complex64)                    # Memory allocation for the eigenvalues
    W  = np.zeros((2*N+1, 2*N+1,Nlt), dtype=np.complex64)         # Memory allocation for the eigenvectors
    W_new  = np.zeros((2*N+1, 2*N+1,Nlt), dtype=np.complex64)         # Memory allocation for the eigenvectors
    Q  = np.zeros((2*N+1, Nlt), dtype=np.complex64)               # Memory allocation for the square root of eigenvalues
    Xi = np.zeros((2*N+1, 2*N+1,Nlt), dtype=np.complex64)         # Memory allocation for the 
    # Tsubi = np.zeros((4*N+2, 4*N+2), dtype=np.complex64)          # Memory allocation for a part of the Tbar matrix 
    # Tsubi1 = np.zeros((4*N+2, 4*N+2), dtype=np.complex64)         # Memory allocation for a part of the Tbar matrix
    # Tbar  = np.zeros((4*N+2, 4*N+2, Nlt-1), dtype=np.complex64)   # Memory allocation for the Tbar matrix used for validation
    # B = np.zeros((2*N+1,2*N+1,Nlt), dtype=np.complex64)         # Memory allocation for matrix B used for error checking
    n = np.arange(-N,N+1)
    Q2check = 1-np.cos(theta_inc_rad)**2-(n*lambda_0/p)**2+2*np.cos(theta_inc_rad)*n*lambda_0/p # Find the Q2 check values for proving they are on the right place
    idxcheck = Q2check.argsort()[::1]                                                           # Find what modes are ordered at what place
    
    for i in range(Nlt):
        Q2, W[:,:,i] = np.linalg.eig(A[:,:,i])                   # Eigenvalues Q2 and eigenvectors W for layer i
        idx = Q2.argsort()[::-1]                                  # Find different different eigenvalues places 
        Q2 = Q2[idx]                                              # Place the eigenvalues from low to high
        W[:,:,i] = W[:,idx,i]                                     # Also replace the eigenvectors
        for j in range(0, 2*N+1):
            Q2_new[idxcheck[j]] = Q2[j]                           # Place Q2 in the same place as the check
            W_new[:,idxcheck[j], i] = W[:, j, i]                  # Place W_new in the same place as the check
            Q[:,i] = np.sqrt(Q2_new)                              # Find the eigenvalues with the sqrt             
        Xi[:,:,i] = np.diag(np.exp(-k_0*Q[:,i]*Hs))               # Find the Xi values of the corresponding Q vectors for each layer  
            
    W = W_new                                                       # Replace 
    Xi[:,:,0] = np.identity(2*N+1)                                  # The X_0 is defined as identity as discussed in report
    Xi[:,:,Nlt-1] = np.identity(2*N+1)                              # The X_0 is defined as identity as discussed in report            
       
        #for loop to allign all values of W to get 1's at diagonal and Q with similar shift 
        # for l in range(2):
        #     for j in range(2*N+1):
        #         if W[j,j,i] != np.max(W[:,j,i]):
        #             for k in range(2*N+1):
        #                 if W[j,k,i] == np.max(W[:,k,i]):
        #                     # if np.max(W[:,k,i])==W[N,k,i]:
        #                         # shift = N-k
        #                     W[:,[j, k],i] = W[:,[k,j],i]
        #                     Q[[j,k],i] = Q[[k,j],i]
        # Xi[:,:,i] = np.diag(np.exp(-k_0*Q[:,i]*Hs))      
        
        # Validation: error checking from diagonalisation checks the difference between A and the matrix constructed back from the eigenvalues and eigenvectors
        # B[:,:,i]=reduce(np.matmul,[W[:,:,i],np.diag(Q[:,i]*Q[:,i]),np.linalg.inv(W[:,:,i])])
        # for j in range(2*N+1):
        #     for k in range(2*N+1):
        #         if np.abs(A[j,k,i]-B[j,k,i])>1/10000000:
        #             print("error",np.abs(A[j,k,i]-B[j,k,i]))
    
    # Xi[:,:,0] = np.identity(2*N+1)                                  # The X_0 is defined as identity as discussed in report
    # Xi[:,:,Nlt-1] = np.identity(2*N+1)                              # The X_0 is defined as identity as discussed in report
    
    for i in range(Nlt-1):
        T_Bar_Mat[:,:,i],  T_BSub1[:,:,i], T_BSub2[:,:,i] = T_Matrix_WQ(W[:,:,i],Q[:,i],W[:,:,i+1],Q[:,i+1],N)
        S_Bar_Matrix[:,:,i] = T_Mat_to_S_Mat(T_Bar_Mat[:,:,i],N)   
        S_Mat_Stable[:,:,i]= S_Mat_Bar_to_S(S_Bar_Matrix[:,:,i], i,Xi,N)   
        # S_Mat_Dir[:,:,i] = S_Mat_Direct(T_Bar_Mat[:,:,i],Xi,i,N) 
        
        T_Matrix[:,:,i], zero,zero = T_Matrix_WQX(W[:,:,i],Q[:,i],W[:,:,i+1],Q[:,i+1],Xi[:,:,i],Xi[:,:,i+1],N)
        # S_Matrix[:,:,i] = T_Mat_to_S_Mat(T_Matrix[:,:,i],N)
    
        
        # S_full = Full_to_Sections(S_Matrix[:,:,i],N)
        S_full_stable = Full_to_Sections( S_Mat_Stable[:,:,i],N)
        S_full = S_full_stable
        #redheffer loop, for the 0th iteration only 1 interface = Smatrix itself
        S = Full_to_Sections(np.identity(4*N+2),N)
        if i == 0:
            S_Redheffd = Redheffer(S_full,S,N)                   # Use Redheffer to compute S0,i+1
            # S_Redheffd_stable = Redheffer(S_full_stable, S)
        else:
            S_Redheffd = Redheffer(S_Redheffd,S_full,N) # Redheffer for Redheffer and new interface
            # S_Redheffd_stable = Redheffer(S_Redheffd_stable,S_full_stable)
            
        S_Redheff_test[:,:,i] = Sections_to_Full(S_Redheffd)  
        # S_RedhefS_test[:,:,i] = Sections_to_Full(S_Redheffd_stable)  
        
    #For debugging purposes:
    # S_Redheffd = S_Redheffd_stable    
    '''
     Check if diagonal or every value is below 1 in all S-Matrices for homogeneous medium
    ''' 
        #debugging if statements
        # if (np.less_equal(np.linalg.norm(S_Matrix[:,:,i]),np.ones((4*N+2, 4*N+2))).all()) == True:
        #     print("This looks like the right way")
        #     Whait = np.less(abs(S_Matrix[:,:,1]),np.ones((4*N+2, 4*N+2)))
        #     Whait2= np.less(abs(test_full),np.ones((4*N+2, 4*N+2)))
        # else:
        #     print("Better luck next time")
        #     # Whait = np.less(abs(S_Matrix[:,:,1]),np.ones((4*N+2, 4*N+2)))
        #     # Whait2= np.less(abs(test_full),np.ones((4*N+2, 4*N+2)))
        
    #These staments should whenever for debugging purposes
    print("t should be less than 1 and it is",np.power(np.linalg.norm(S_Redheffd[0,:,N]),2))
    print("r should be less than 1 and it is",np.power(np.linalg.norm(S_Redheffd[2,:,N]),2))
    # print("The rest of the energy is ",np.power(np.linalg.norm(S_Redheffd[1,:,N]),2)+np.power(np.linalg.norm(S_Redheffd[3,:,N]),2))
    print("Summing should equal 1 and it is",np.power(np.linalg.norm(S_Redheffd[2,:,N]),2)+np.power(np.linalg.norm(S_Redheffd[0,:,N]),2))
    #  np.less(S_Matrix[:,:,i],np.eye(New.shape[0])) == True:
    
    ''' 
    Now that all Redheffers are calculated the transmission and
    reflection coefficients can be calculated. This will be done in this part!
    '''
    #Memory allocation for c_tranmitted and reflected
    c_transmitted = np.zeros((2*N+1,Nlt-1),dtype =np.complex64)
    c_reflected   = np.zeros((2*N+1,Nlt-1),dtype =np.complex64)
    
    # c_test_min = np.zeros((2*N+1,Nlt-1),dtype =np.complex64)    #Memory c_test_min
    # c_test_plus = np.zeros((2*N+1,Nlt-1),dtype =np.complex64)   #Memory c_test_plus
    
    
    #for the amount of interfaces calculate C_transmitted and reflected
    for i in range(Nlt-1):
        
        if i == 0:
            c_transmitted[N, i] = 1                                             #Defines our input signal
            c_transmitted[:,Nlt-2] = np.matmul(S_Redheffd[0],c_transmitted[:,i])#Defines our transmitted signal, lower layer
            c_reflected[:,i] = np.matmul(S_Redheffd[2],c_transmitted[:,i])      #Defines our reflected signal upper layer
            
            # c_test_plus[:,Nlt-2] = c_transmitted[:,Nlt-2]                       #transmitted signal test
            # c_test_min[:,Nlt-2] = np.zeros((2*N+1),dtype = np.complex64)        #Reflected signal test
            
            
        else:
            T_Sections = Full_to_Sections(T_Matrix[:,:,i-1],N)
            c_transmitted[:,i]  = np.matmul(T_Sections[0],c_transmitted[:,i-1]) + np.matmul(T_Sections[1],c_reflected[:,i-1]) #Calculates next transmission layer 
            c_reflected[:,i]    = np.matmul(T_Sections[2],c_transmitted[:,i-1]) + np.matmul(T_Sections[3],c_reflected[:,i-1]) #Calculates next reflection layer
            
            # T_Sections = Full_to_Sections(T_Matrix[:,:,Nlt-2-i],N)
            #Intermediate terms to bakcwards calculate c_min and c_plus
            # Division_term = np.ones((2*N+1,2*N+1),dtype = np.complex64)+reduce(np.matmul,[np.linalg.inv(T_Sections[3]), T_Sections[2],np.linalg.inv(T_Sections[0]),T_Sections[1]])
            # Multi_term = reduce(np.matmul, [np.linalg.inv(T_Sections[3]),T_Sections[2],np.linalg.inv(T_Sections[0])])
    
            # c_test_min[:,Nlt-2-i] = np.matmul(np.matmul(np.linalg.inv(Division_term),(np.linalg.inv(T_Sections[3]))),c_test_min[:,Nlt-i-1]) - \
            #                         np.matmul(np.matmul(np.linalg.inv(Division_term),Multi_term),c_test_plus[:,Nlt-i-1])
            # c_test_plus[:,Nlt-2-i] = np.matmul(np.linalg.inv(T_Sections[0]),c_test_plus[:,Nlt-i-1]) - reduce(np.matmul,[np.linalg.inv(T_Sections[0]), T_Sections[1], c_test_min[:,Nlt-2-i]])
    
        # if (np.less_equal((np.linalg.norm(c_transmitted[:,i])**2+np.linalg.norm(c_reflected[:,i-1])**2),(np.linalg.norm(c_transmitted[:,i-1])**2+np.linalg.norm(c_reflected[:,i])**2)).all()) == True:
        #     print("Urmegurd, you did it!")
        # else:
        #     print("This interface causes issues:",i)
    if (np.less_equal((np.linalg.norm(c_transmitted[:,Nlt-2])**2+np.linalg.norm(c_reflected[:,0])**2),np.linalg.norm(c_transmitted[:,0])**2).all()) == True:
        print("The C after Redheffer is correct")
    
    Efield = EVisual(c_reflected[:,0],c_transmitted[:,Nlt-2],c_transmitted,c_reflected,z_start,z_stop,k_0,Kx,H,W,Q,accuracy,p,N,e0,e2,Nl) 
    z_start = -0.5*H         
    z   = np.arange(start = z_start, stop = z_stop, step = (z_stop-z_start)/(accuracy))   #Defines the z axis
    x   = np.arange(start = -p/2,stop = p/2, step = p/100)           #Defines the x axis
    
    heatmapEfield(np.imag(Efield),x,H,z_start,z_stop,z,Sl,Sr,Nl,p)
    return