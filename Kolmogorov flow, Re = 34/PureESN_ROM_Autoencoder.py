#%%

"""
Pure deep learning model architecture for predictions on high dimensional systems by Jochem Veerman 
for his Msc thesis project on hybrid knowledge-based/deep learning reduced order modelling 
for high dimensional chaotic systen. The code is inspired by and based on the codes from:

Lesjak and Doan, 2021: 
Chaotic systems learning with hybrid echo state network/proper orthogonal decomposition based model

The general structure and code for the echo state network.
    
and

Racca et al., 2022
Modelling spatiotemporal turbulent dynamics with the convolutional autoencoder echo state network

The general structure and code for the autoencoder
"""

from os import environ
environ['OMP_NUM_THREADS'] = "1"


#%%

import os
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy.linalg as LA
import argparse
import tensorflow as tf
import h5py
tf.get_logger().setLevel('ERROR') #no info and warnings are printed 
tf.config.set_visible_devices([], 'GPU')
#import ESN
import importlib.util
import ESN as EchoStateNet
import matplotlib.animation as animation


#%%

# Functions

def decoder(inputs, dec_mods, is_train=False):
    
    '''
    Multiscale decoder as obtained from the code made by Racca et al., 2022.
    The contribution of each decoder are summed.
    '''
            
    decoded = 0
    for dec_mod in dec_mods:
        decoded += dec_mod(inputs, training=is_train)
        
    return decoded

class PerPad2D(tf.keras.layers.Layer):
    
    """
    Periodic Padding layer, as obtained from the code made by Racca et al., 2022.
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
    Obtained from the code made by Racca et al., 2022.
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

# Even more functions

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

#setup
epsilon = 0.4 # Threshold for the error
normalization = True # Normalize data for echo state network
plot = True # Make plot

# Read data from files
file =  np.load("Folder_of_AE/AE_3C_3L_e12_24_1_d24_12_6_N8_8_1/Encoded_data_Re34.npz")
path = 'Folder_of_AE/AE_3C_3L_e12_24_1_d24_12_6_N8_8_1/Kflow_RE34_AE' # To extract autoencder

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

X_enc = X_enc[:90000] # Select first part of input data

# Read reference data
file_ref = np.load("Reduced_Kolmogorov_Re34.npz")
X_ref = file_ref["X"]

gridpoints = len(coord) # Number of gridpoints
nk = int((gridpoints - 1)/2) # Amount of wavenumbers

#%%

#ESN setup
K.clear_session()
rng = np.random.RandomState(1) # Fixed seed
n_neurons = 3000 # Number of neurons in the resevoir
degree = 3.5  # average number of connections of a unit to other units in reservoir (3.0)
input_scaling = 0.01 # Input scaling
sparseness = 1. - degree / (n_neurons - 1.)  # sparseness of W
rho = 5.0 # Spectral radius
leaking_rate = 0.33 # Leaking rate
beta = 0.0001 # Tikhonov factor 

#%%

# Training and validation settings
upsample = 1 # Increase delta t by a factor of upsample to decrease computational time
dt = dt*upsample
X_enc = X_enc[::upsample] # Skip every upsample-th data
X_ref = X_ref[::upsample]

T_lyap = 1/Lyap # Lyapunov time
washout   = 100 # Washout for ESN training
t_train   = 500*T_lyap # Training time for ESN
N_train   = int(t_train/dt) # Train certain number of Lyapunov times
t_stop    = int(40*T_lyap) # Predict certain number of Lyapunov times
tt = np.arange(0,t_stop+dt,dt)/T_lyap # time array, per measurement

idx_start = 74626 # Initial conditions of this run 

#%%

# Normalize data
stddev = np.std(X_enc, axis=0)
avg = np.average(X_enc, axis=0)

if normalization == True:
    X_enc = (X_enc - avg) / stddev

input_all_enc = X_enc[:-1, :] # Input data containing all time steps except the last
output_all_enc = X_enc[1:, :] # Output data containing all time steps except first

# Obtain training data
input_train_enc = input_all_enc[:N_train, :] 
output_train_enc = output_all_enc[:N_train, :]

#%% 

# Initilization time and washout for the ESN
resync_time = 20
resync_idx = int(resync_time/(Lyap*dt))

input_train_enc = input_train_enc.reshape((1,input_train_enc.shape[0],input_train_enc.shape[1]))

#%%

# Configure the model
model = EchoStateNet.ESN(n_neurons, input_train_enc.shape[2], leaking_rate, rho, sparseness, input_scaling, rng, 'tanh', beta)

# Train model
model.fit(input_train_enc, output_train_enc, washout)
model.reset_states()

print("model with: " + str(n_neurons) + " neurons, " + str(sparseness) + " sparseness, " +
           str(leaking_rate) + " leaking_rate, " + str(rho) + " rho, " + str(input_scaling) + " input_scaling, "
           + str(beta) + " beta")

# Teference solution for prediction section
x_ref_test = X_ref[:, :]
x_enc_test = X_enc[:, :]

#%%

# Initial data set to initialize the reservor
# Here the reservoir is initialized with a certain number of the snapshots from the training/validation dataset
x_init = X_enc[idx_start - resync_idx: idx_start, :]

# Local reference solution
x_ref_loc = x_ref_test[idx_start:idx_start+len(tt), :]  
x_enc_loc = x_enc_test[idx_start:idx_start+len(tt), :]

# Initialize reservoir
x_init = x_init.reshape((1, x_init.shape[0], x_init.shape[1]))
model.predict(x_init) # Perform prediction using the model to initialze it

# List of predicted solutions
x_pred = [] 

y_last = x_enc_loc[0, :].reshape((1, 1, x_enc_loc.shape[1]))
x_pred.append(x_enc_loc[0, :].reshape((1, x_enc_loc.shape[1])))

# Time predictions based on output of the previous timestep
print('start natural response') 
for i in range(len(tt) - 1):
    x_pred.append(model.predict(y_last))
    y_last = x_pred[-1].reshape((1, 1, x_enc_loc.shape[1]))
    print('[%d/%d] predicitions done' % ((i), x_enc_loc.shape[0])) if i % 100 == 0 else None

model.reset_states() # Reset states of ESN

#%%                                    
       
x_pred = np.array(x_pred)
x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[2])
    
# denormalize
if normalization == True:
        x_pred = x_pred * stddev + avg

#%%

# Decode the data

# Import the decoder
dec = [None]*(int(N_parallel))
for i in range(int(N_parallel)):
    dec[i] = tf.keras.models.load_model(path + '/dec_mod'+str(tuple(ker_size[i]))+'_'+str(N_latent)+'.h5',
                                            custom_objects={"PerPad2D": PerPad2D})

# Back to the 2D matrix format for the decoder
U_enc_pred = np.zeros((x_pred.shape[0], N_1[1], N_1[2], N_1[3]))
U_enc_pred = X_to_U_enc(x_pred, N_1)

# Decode the solution
U_dec_pred = np.zeros((x_pred.shape[0], len(coord), len(coord), 2))

N_pos         = 23 #split in k interval of N_pos length needed to process long timeseries
k             = int(x_pred.shape[0]/N_pos)

for i in range(k):
    U_dec_pred[i*N_pos:(i+1)*N_pos]= decoder(U_enc_pred[i*N_pos:(i+1)*N_pos], dec)
   
# Decode rest of the solution that is outside the k*N_pos     
U_dec_pred[k*N_pos:]= decoder(U_enc_pred[k*N_pos:], dec)

# Reshape into 1D format to compare with reference solution
x_pred_decoded = np.zeros((U_dec_pred.shape[0],2*U_dec_pred.shape[1]*U_dec_pred.shape[2]))
x_pred_decoded = U_to_X_dec(U_dec_pred, len(coord))

#%%

# calculate Error
err = LA.norm(x_pred_decoded - x_ref_loc, axis=1) / np.sqrt(
    np.average(np.square(LA.norm(x_ref_loc, axis=1))))

if plot == True:

    # calculate Error
    plt.plot(tt,err)
    plt.axvline((np.argmax(err > epsilon) + 1) * dt/T_lyap,0,1,label='valid time',color='red')
    plt.xlabel("t")
    plt.ylabel("RMSE")
    plt.title("Prediction horizon ESN, reservoir = %s" %n_neurons)
    plt.show()