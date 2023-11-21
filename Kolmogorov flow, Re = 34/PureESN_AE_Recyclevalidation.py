#%%

"""
Hybrid model architecture for predictions on high dimensional systems by Jochem Veerman 
for his Msc thesis project on hybrid knowledge-based/deep learning reduced order modelling 
for high dimensional chaotic systen. The code is inspired by and based on the codes from:

Lesjak and Doan, 2021: 
Chaotic systems learning with hybrid echo state network/proper orthogonal decomposition based model

The general structure and code for the echo state network and the proper orthogonal decomposition
and Galerkin projection
    
and

Racca et al., 2022
Modelling spatiotemporal turbulent dynamics with the convolutional autoencoder echo state network

The general structure and code for the autoencoder

and 

A. Racca and L. Magri. Robust optimization and validation of echo state networks for
learning chaotic dynamics. Neural Networks, 142:252–268, 2021

for the recycle validation
"""

# Import modules 

from os import environ
environ['OMP_NUM_THREADS'] = "1"

import os
import numpy.linalg as LA
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
tf.get_logger().setLevel('ERROR') #no info and warnings are printed 
tf.config.set_visible_devices([], 'GPU')
import matplotlib.pyplot as plt
import ESN_paramsearch as EchoStateNet
import time
import skopt
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import Matern, WhiteKernel, Product, ConstantKernel
from skopt.plots import plot_convergence

#%%

# Functions for the Auto Encoder

def decoder(inputs, dec_mods, is_train=False):
    
    '''
    Multiscale decoder, taken from Hasegawa 2020. The contribution of the CNNs at different
    scales are simply summed.
    '''
            
    decoded = 0
    for dec_mod in dec_mods:
        decoded += dec_mod(inputs, training=is_train)
        
    return decoded

class PerPad2D(tf.keras.layers.Layer):
    """
    Periodic Padding layer
    """
    def __init__(self, padding=1, asym=False, **kwargs):
        self.padding = padding
        self.asym    = asym
        super(PerPad2D, self).__init__(**kwargs)
        
    def get_config(self): #needed to be able to save and load the model with this layer
        config = super(PerPad2D, self).get_config()
        config.update({
            'padding': self.padding,
            'asym': self.asym,
        })
        return config

    def call(self, x):
        return periodic_padding(x, self.padding, self.asym)
    
def periodic_padding(image, padding=1, asym=False):
    '''
    Create a periodic padding (same of np.pad('wrap')) around the image, 
    to mimic periodic boundary conditions.
    When asym=True on the right and lower edges an additional column/row is added
    '''
        
    if asym:
        lower_pad = image[:,:padding+1,:]
    else:
        lower_pad = image[:,:padding,:]
    
    if padding != 0:
        upper_pad     = image[:,-padding:,:]
        partial_image = tf.concat([upper_pad, image, lower_pad], axis=1)
    else:
        partial_image = tf.concat([image, lower_pad], axis=1)
        
    if asym:
        right_pad = partial_image[:,:,:padding+1] 
    else:
        right_pad = partial_image[:,:,:padding]
    
    if padding != 0:
        left_pad = partial_image[:,:,-padding:]
        padded_image = tf.concat([left_pad, partial_image, right_pad], axis=2)
    else:
        padded_image = tf.concat([partial_image, right_pad], axis=2)

    return padded_image

#%%


def U_to_X_dec(U_input, length):
    
    """
    Function to flatten the 2D matrices into the 1D format for the ESN
    """
    
    X_flattened = np.zeros((U_input.shape[0], 2*length*length))
    
    for t in range(U_input.shape[0]):
     for j in range(U_input.shape[1]):
         for i in range(U_input.shape[2]):
             X_flattened[t,i+j*U_input.shape[2]]=U_input[t,j,i,0]
             X_flattened[t,U_input.shape[1]*U_input.shape[2]+i+j*U_input.shape[2]]=U_input[t,j,i,1]
     
    return X_flattened

def X_to_U_enc(X_flattend, N_1): # Only works for Nx = Ny

    """
    Function to unflatten the 1D data format back into the 2D matrices for each time step
    Function works only in latent space
    """
    
    U_2D = np.zeros((X_flattend.shape[0], N_1[1], N_1[2], N_1[3]))  
    
    for k in range(N_1[3]):
        
        Xu = X_flattend[:, k*N_1[1]*N_1[2] : k*N_1[1]*N_1[2] + N_1[1]*N_1[2] ]
        
        for j in range(X_flattend.shape[0]):
            for i in range(N_1[2]):
                U_2D[j, i, : , k] = Xu[j, N_1[1]*i : N_1[1]*i + N_1[1]]

    return U_2D

#%%

# Function and class definition

#Hyperparameter Optimization using Grid Search plus Bayesian Optimization

def g(val):
    
    #Gaussian Process reconstruction
    b_e = GPR(kernel = kernell,
            normalize_y = True, #if true mean assumed to be equal to the average of the obj function data, otherwise =0
            n_restarts_optimizer = 3,  #number of random starts to find the gaussian process hyperparameters
            noise = 1e-10, # only for numerical stability
            random_state = 1) # seed    
    
    #Bayesian Optimization
    res = skopt.gp_minimize(val,                         # the function to minimize
                      search_space,                      # the bounds on each dimension of x
                      base_estimator       = b_e,        # GP kernel
                      acq_func             = "EI",       # the acquisition function
                      n_calls              = n_tot,      # total number of evaluations of f
                      x0                   = x1,         # Initial grid search points to be evaluated at
                      n_initial_points     = n_in,       # the number of additional random initialization points
                      n_restarts_optimizer = 3,          # number of tries for each acquisition
                      random_state         = 1,         # seed
                           )   
    return res


def RCV_Noise(x): 
    #Recycle Validation
    
    global best_opt, m

    #setting and initializingx
    rho      = np.round(x[0], 2)
    input_scaling = np.round(x[1], 2)
    leaking_rate = np.round(x[2],2)
    mean_per_beta = np.zeros(len(beta_array))
    

    for k in range(len(beta_array)):
        
        K.clear_session() # Start a new tensorflow session
        rng = np.random.RandomState(1)
        
        model = EchoStateNet.ESN(n_neurons, input_train_enc.shape[2], leaking_rate, rho, sparseness, input_scaling, rng, 'tanh')

        beta = beta_array[k]
        print(beta)
        #train model
        
        model.fit(input_train_enc, output_train_enc, washout, beta)
        model.reset_states()
        
        horizon = []
        
        for i in range(amount_of_runs): # Loop over 20 runs for the statistics
    
            idx_start = int(i*IC_timeinterval/dt) + resync_idx # IC of this run
            print("idx_start = ", idx_start)

            # Initilize the ESN
            x_init = X_enc[idx_start - resync_idx: idx_start, :]
    
            # local reference solution
            x_ref_loc = x_ref_test[idx_start:idx_start+len(tt), :]  
            x_enc_loc = x_enc_test[idx_start:idx_start+len(tt), :]
                
            # initialize reservoir
            x_init = x_init.reshape((1, x_init.shape[0], x_init.shape[1]))
            model.predict(x_init)
            
            # Empty list to fill with solutions
            Y = []
        
            # Take the first value of the local solution as start point for the ESN
            y_last = x_enc_loc[0, :].reshape((1, 1, x_enc_loc.shape[1]))
            Y.append(x_enc_loc[0, :].reshape((1, x_enc_loc.shape[1])))
        
            # Produce temporal predictions with the ESN
        
            #print('start natural response')
            for j in range(tt.shape[0] - 1):
                Y.append(model.predict(y_last))
                y_last = Y[-1].reshape((1, 1, x_enc_loc.shape[1]))
                #print('[%d/%d] predicitions done' % ((j), tt.shape[0])) if j % 100 == 0 else None
        
            # Reset states
            model.reset_states()
        
            # Convert list into an array and reshape into known format
            Y = np.array(Y)
            Y = Y.reshape( Y.shape[0], Y.shape[2])

            # denormalize
            if normalization == True:
                Y = Y * stddev + avg
                  
            # Import the decoder
            dec = [None]*(int(N_parallel))
            for s in range(int(N_parallel)):
                dec[s] = tf.keras.models.load_model(path + '/dec_mod'+str(tuple(ker_size[s]))+'_'+str(N_latent)+'.h5',
                                                        custom_objects={"PerPad2D": PerPad2D})
            # Bring back to the 2D matrix format
            
            U_enc_pred = np.zeros((Y.shape[0], N_1[2], N_1[2], N_1[3]))
            U_dec_pred = np.zeros((Y.shape[0], len(coord), len(coord), 2))
            U_enc_pred = X_to_U_enc(Y, N_1)
                      
            N_pos         = 23 #split in k interval of N_pos length needed to process long timeseries
            count         = int(U_enc_pred.shape[0]/N_pos)
            
            for s in range(count):
                U_dec_pred[s*N_pos:(s+1)*N_pos]= decoder(U_enc_pred[s*N_pos:(s+1)*N_pos], dec)
               
            # Decode rest of the solution that is outside the k*N_pos     
            U_dec_pred[count*N_pos:]= decoder(U_enc_pred[count*N_pos:], dec)

            # Reshape into 1D format to compare with reference solution
            x_pred_decoded = np.zeros((U_dec_pred.shape[0],2*U_dec_pred.shape[1]*U_dec_pred.shape[2]))
            x_pred_decoded = U_to_X_dec(U_dec_pred, len(coord))
            
            # calculate Error
            err = np.linalg.norm(x_pred_decoded - x_ref_loc, axis=1) / np.sqrt(
                np.average(np.square(np.linalg.norm(x_ref_loc, axis=1))))
        
            # Error is computed, take time where error is exceeded for the first time and store
            t = (np.argmax(err > epsilon) + 1) * dt/T_lyap

            if any(err > epsilon) == False:
                t = t_stop/T_lyap
                print("Prediction time exceeds N_val")

            horizon.append(t) # This array thus becomes amount_of_runs long
            print(horizon)
            
        mean_per_beta[k] = np.mean(horizon)
        
    best_beta_idx = np.argmax(mean_per_beta)
    beta_opt[m] = beta_array[best_beta_idx]
    best_mean = mean_per_beta[best_beta_idx] 
    m += 1
    
    return 1/best_mean
        

#%%

#setup
epsilon = 0.4 # Threshold error
normalization = True

# Read data from files
file =  np.load("../../../Folder_of_AE/AE_3C_3L_e12_24_1_d24_12_6_N8_8_1/Encoded_data_Re34.npz")
path = '../../../Folder_of_AE/AE_3C_3L_e12_24_1_d24_12_6_N8_8_1/Kflow_RE34_AE' # To extract autoencder

Re = file["Re"] # Reynolds number
Lyap = 0.067 # Lyapunov exponent
dt = file["dt"] # Time step of the data sampling
coord = file["coord"]
X_enc = file["U_enc"] # Encoded data
n_epochs = file["n_epochs"] # Number of epoch during training of the autoencoder
N_parallel = file["N_parallel"] # Number of encoders/decoders
ker_size = file["ker_size"] # Kernel sizes used in the encoders/decoders
N_latent = file["N_latent"] # Latent space size
N_1 = file["N_1"] # Laten space dimensions

X_enc = X_enc[:90000]

# Read reference data
file_ref =np.load("../../../Reduced_Kolmogorov_Re34.npz")
X_ref = file_ref["X"]

gridpoints = len(coord)
nk = int((gridpoints - 1)/2)
data_len = X_enc.shape[0]

#%%

# Training and validation settings
upsample = 1 # Increase delta t by a factor of upsample to decrease computational time
dt = dt*upsample # if we gonna skip every upsample, dt becomes a factor upsample bigger
X_enc = X_enc[::upsample] # Skip every upsample-th data
X_ref = X_ref[::upsample]

T_lyap = 1/Lyap # Lyapunov time
washout   = 100
t_train   = 500*T_lyap
N_train   = int(t_train/dt) # Train certain number of lyapunov times
t_stop    = int(20*T_lyap) # Predict certain number of lyapunov times
tt = np.arange(0,t_stop+dt,dt)/T_lyap # time array, per measurement

# Resync time for the ESN to burn in
resync_time = 20*T_lyap
resync_idx = int(resync_time/dt)

amount_of_runs = 2
potential_time_lastIC = t_train - t_stop - resync_time # Last step of the training time minus 1 prediction set
IC_timeinterval = potential_time_lastIC/amount_of_runs  # Amount of timestep between IC_snapshots

#%%

# normalize data
stddev = np.std(X_enc, axis=0)
avg = np.average(X_enc, axis=0)

if normalization == True:
    X_enc = (X_enc - avg) / stddev
    
input_all_enc = X_enc[:-1, :] # Input data containing all time steps except the last
output_all_enc = X_enc[1:, :] # Output data containing all time steps except first, so "1 step ahead" of input data

# Get training data using the idx_split index. So not that there seems to be no validation data
input_train_enc = input_all_enc[:N_train, :] 
output_train_enc = output_all_enc[:N_train, :]

input_train_enc = input_train_enc.reshape((1,input_train_enc.shape[0],input_train_enc.shape[1]))

#reference solution for prediction section
x_ref_test = X_ref[:, :]
x_enc_test = X_enc[:, :]

#%%

# Baysian optimisation and grid search preparation
n_neurons = 500
degree = 3.5 
sparseness = 1. - degree / (n_neurons - 1.)  # sparseness of W

rho_in     = 0.01    #range for hyperparameters (spectral radius and input scaling)
rho_end    = 5.0 
input_scaling_in  = 0.01
input_scaling_end = 5.0
leaking_rate_in = 0.1
leaking_rate_end = 1.0

n_in  = 7        #Number of Initial random points, set to zero because we input an explicit grid of point for rho and inputscaling to evaluate (see next)

# In case we want to start from a grid_search, the first n_grid_x*n_grid_y points are from grid search
n_grid_x = 0 # if zero, the grid search is turned off
n_grid_y = n_grid_x
n_grid_z = n_grid_x
n_bo     = 7  #number of points to be acquired through BO after grid search
n_tot    = n_grid_x*n_grid_y*n_grid_z + n_bo + n_in #Total Number of Function Evaluatuions

# computing the points in the grid
if n_grid_x > 0:
    x1    = [[None] * 3 for i in range(n_grid_x*n_grid_y*n_grid_z)]
    count     = 0
    for i in range(n_grid_x):
        for j in range(n_grid_y):
            for k in range(n_grid_z):
                x1[count] = [rho_in + (rho_end - rho_in)/(n_grid_x-1)*i,
                             input_scaling_in + (input_scaling_end - input_scaling_in)/(n_grid_y-1)*j,
                             leaking_rate_in + (leaking_rate_end - leaking_rate_in)/(n_grid_z-1)*k]
                count   += 1
                    
else:
    x1 = None
    
beta_array = np.array([1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001])
beta_opt = np.zeros(n_tot)
                    
  
# range for hyperparameters, can be more than 2 but note that also the initial grid should be adapted accordingly!
search_space = [Real(rho_in, rho_end, name='spectral_radius'),
                Real(input_scaling_in, input_scaling_end, name='input_scaling'),
                Real(leaking_rate_in, leaking_rate_end, name='leaking_rate')]

# ARD 5/2 Matern Kernel with sigma_f in front for the Gaussian Process
kernell = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-1, 3e0))*\
                  Matern(length_scale=[0.2,0.2, 0.1], nu=2.5, length_scale_bounds=(5e-2, 1e1))  
                  
#%%

# Hyperparameter search set-up

#Number of Networks in the ensemble
ensemble = 1

# Which validation strategy (implemented in Val_Functions.ipynb)
val      = RCV_Noise                 # validation method

#Quantities to be saved
par      = np.zeros((ensemble, 4))      # GP parameters
x_iters  = np.zeros((ensemble,n_tot,3)) # coordinates in hp space where f has been evaluated
f_iters  = np.zeros((ensemble,n_tot))   # values of f at those coordinates
minimum  = np.zeros((ensemble, 5))      # minima found per each member of the ensemble

# save the final gp reconstruction for each network
gps        = [None]*ensemble

#%%

# BO hyperparameter search

# optimize ensemble networks (to account for the random initialization of the input and state matrices)
for i in range(ensemble): # Loop over all ESN in the ensemble
    
    print('Realization    :',i+1)
    
    m   = 0
    
    # Bayesian Optimization
    res      = g(val) # Perform the baysian optimisation
    
    
    #Saving Quantities for post_processing
    gps[i]     = res.models[-1]    
    gp         = gps[i]
    x_iters[i] = np.array(res.x_iters)
    f_iters[i] = np.array(res.func_vals)
    minimum[i] = np.append(res.x, [beta_opt[np.argmin(f_iters[i])], res.fun])
    params     = gp.kernel_.get_params()
    key        = sorted(params)
    par[i]     = np.array([params[key[2]],params[key[5]][0], params[key[5]][1], gp.noise_])
    
    
    #Plotting Optimization Convergence for each network
    print('Best Results: ' + 'rho = ', minimum[i,0], 'input_scaling = ', minimum[i,1], 'leaking_rate = ', minimum[i,2], "beta = ", minimum[i,3],
              '1/(Max mean horizon):', minimum[i,-1])
    plt.rcParams["figure.figsize"] = (15,2)
    plt.figure()
    plot_convergence(res)
    plt.show()























