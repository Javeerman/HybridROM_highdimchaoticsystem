# Import modules 

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

from os import environ
environ['OMP_NUM_THREADS'] = "1"


import numpy.linalg as LA
import numpy as np
import os
from scipy.fft import fftfreq, fft2, ifft2
import tensorflow as tf
from tensorflow.keras import backend as K
tf.get_logger().setLevel('ERROR') #no info and warnings are printed 
tf.config.set_visible_devices([], 'GPU')
import matplotlib.pyplot as plt
import ESN_Hybrid_paramsearch as EchoStateNet
import skopt
from skopt.space import Real
from skopt.learning import GaussianProcessRegressor as GPR
from skopt.learning.gaussian_process.kernels import Matern, WhiteKernel, Product, ConstantKernel
from skopt.plots import plot_convergence


#%%

# Functions for PODGL

class POD:

    def __call__(self, input, normalize):
        """Proper Orthogonal Decomposition

        :param input: 2D array (m x n) m: Number of Timesteps, n: Number of Variables
        :param normalize: Boolean True or False, whether to substract mean from each row
        :return: eig: Eigenvalues of autocorrelation matrix, a: temporal coefficients, phi: POD modes
        """
        if normalize == True:
            self.avg = np.average(input, axis=0)
            input = input - self.avg

        #number of timesteps
        m = input.shape[0]
        C = np.matmul(np.transpose(input),input)/(m-1)

        #solve eigenvalue problem
        eig, phi = LA.eigh(C)

        #Sort Eigenvalues and vectors
        idx = eig.argsort()[::-1]
        eig = eig[idx]
        phi = phi[:, idx]
        if normalize == True:
            self.avg = self.avg[idx]

        #project onto modes for temporal coefficients
        a = np.matmul(input,phi)

        return eig, a, phi

    def denorm(self,input):
        #check whether input has been reduced
        dim_in = input.shape[1]

        return input + self.avg[0:dim_in]

class ROM():

    def __init__(self,phi,coord,h,Re,fln=None):


        self.N = int((coord.shape[0]-1)/2)
        self.n_mod = phi.shape[1]
        n = 4
        self.Re = Re

        # physical grid
        self.dx = coord[1]-coord[0]
        xx, yy = np.meshgrid(coord, coord)
        fx = np.sin(n * yy)

        self.phi = phi
        #roll arrays
        self.phi_roll = self.roll(phi)

        self.fx_trfd = self.proj_fx(fx)

        if os.path.exists(fln) == False:
            print('No File found for given number of skipped modes')
            self.calc_der()
            self.calc_params(fln)
        else:
            print('File found, load parameters')
            file = np.load(fln)
            self.A = file['A']
            self.B = file['B']



        #ETDRK4 params
        self.h = h
        w, v = LA.eig(self.A)
        inv_v = LA.inv(v)

        E = np.exp(h * w)
        self.E = np.matmul(v, np.matmul(np.diag(E), inv_v))
        E_2 = np.exp(h * w / 2)
        self.E_2 = np.matmul(v, np.matmul(np.diag(E_2), inv_v))
        Q = 0
        phi1 = 0.0
        phi2 = 0.0
        phi3 = 0.0

        M = 32  # number of evaluation points in Cauchy integral

        for j in range(1, M + 1):
            arg = h * w + np.ones(w.shape[0]) * np.exp(2j * np.pi * (j - 0.5) / M)

            phi1 += 1.0 / arg * (np.exp(arg) - np.ones(w.shape[0]))
            phi2 += 1.0 / arg ** 2 * (np.exp(arg) - arg - np.ones(w.shape[0]))
            phi3 += 1.0 / arg ** 3 * (np.exp(arg) - 0.5 * arg ** 2 - arg - np.ones(w.shape[0]))
            Q += 2.0 / arg * (np.exp(0.5 * arg) - np.ones(w.shape[0]))

        phi1 = np.real(phi1 / M)
        phi1 = np.matmul(v, np.matmul(np.diag(phi1), inv_v))
        phi2 = np.real(phi2 / M)
        phi2 = np.matmul(v, np.matmul(np.diag(phi2), inv_v))
        phi3 = np.real(phi3 / M)
        phi3 = np.matmul(v, np.matmul(np.diag(phi3), inv_v))
        Q = np.real(Q / M)
        self.Q = np.matmul(v, np.matmul(np.diag(Q), inv_v))

        self.f1 = phi1 - 3 * phi2 + 4 * phi3
        self.f2 = 2 * phi2 - 4 * phi3
        self.f3 = -phi2 + 4 * phi3




    def NonLin(self, a):

        temp = np.tensordot(self.B,a, axes=([2,0]))
        temp = np.tensordot(a,temp, axes=([0,1]))

        return temp + self.fx_trfd

    def proj_fx(self, fx):

        fx_trfd = np.zeros((self.n_mod))

        for mode in range(self.n_mod):

            fx_trfd[mode] = np.sum(self.phi_roll[0,mode,:,:] * fx)

        return fx_trfd


    def integrate(self,u):

        #assert x.size == 3, 'Input needs 3 entries'

        Nu = self.NonLin(u)
        a = np.matmul(self.E_2, u) + self.h / 2 * np.matmul(self.Q, Nu)
        Na = self.NonLin(a)
        b = np.matmul(self.E_2, u) + self.h / 2 * np.matmul(self.Q, Na)
        Nb = self.NonLin(b)
        c = np.matmul(self.E_2, a) + self.h / 2 * np.matmul(self.Q, 2*Nb-Nu)
        Nc = self.NonLin(c)
        # update rule
        u = np.matmul(self.E,u) + self.h * np.matmul(self.f1, Nu) + self.h * np.matmul(self.f2, Na+Nb) + self.h * np.matmul(self.f3,Nc)

        return u


    def roll(self,phi):

        phi_new = np.zeros((2, self.n_mod, len(coord), len(coord)))
        
        """Okee so we roll the given mode matrix and we do it as follows in a 4D matrix:
        We select a single mode n. And look at the u and v component of the flow seperatly.
        and for each of these, we consider a rectangle of 2N+1 by 2N+1 indices. Now we loop over
        the indices j and i. The idea here is to obtain a 2D picture of the eigenmodes. So for v = 0 
        corresponding to the u-component, we go (2N+1) times over (2N+1) values starting from the first index
        and thus we filter out all u_modes. For the v-component, we start at (2N+1)^2 and again do the same, 
        filtering out the v-components. This way u, v are seperated from each other and stored onto a
        2D matrix fahsion"""

        for n in range(self.n_mod):
            for v in range(2):
                for j in range(len(coord)):
                    for i in range(len(coord)):
                        phi_new[v, n, j, i] = phi[i+(len(coord))*j+(len(coord))*(len(coord))*v, n]

        print('(%d/%d) steps done' % (n, self.n_mod)) if n % 100 == 0 and n != 0 else None


        return phi_new


    def calc_params(self,fln):

        self.B = np.zeros((self.n_mod,self.n_mod,self.n_mod))
        self.A = np.zeros((self.n_mod,self.n_mod))

        shear = 0.5*(self.phi_roll_x + np.transpose(self.phi_roll_x,axes=(4,1,2,3,0)))

        for k in range(self.n_mod):
            for m in range(self.n_mod):

                self.A[k,m] = np.sum(self.phi_roll_x[0,k,:,:,0] * shear[0,m,:,:,0])
                self.A[k,m] += np.sum(self.phi_roll_x[0,k,:,:,1] * shear[0,m,:,:,1])
                self.A[k,m] += np.sum(self.phi_roll_x[1,k,:,:,0] * shear[1,m,:,:,0])
                self.A[k,m] += np.sum(self.phi_roll_x[1,k,:,:,1] * shear[1,m,:,:,1])
                self.A[k,m] = -2/self.Re * self.A[k,m]

                for n in range(self.n_mod):

                    temp1 = self.phi_roll[0,m,:,:] * self.phi_roll_x[0,n,:,:,0] + self.phi_roll[1,m,:,:] * self.phi_roll_x[0,n,:,:,1]
                    temp2 = self.phi_roll[0,m,:,:] * self.phi_roll_x[1,n,:,:,0] + self.phi_roll[1,m,:,:] * self.phi_roll_x[1,n,:,:,1]
                    self.B[k,m,n] = -np.sum(self.phi_roll[0,k,:,:] * temp1)-np.sum(self.phi_roll[1,k,:,:] * temp2)

                    print('(%d/%d) in total' % ((k*self.n_mod*self.n_mod)+(m*self.n_mod)+n, self.n_mod**3)) if ((k*self.n_mod*self.n_mod)+(m*self.n_mod)+n) % 100000 == 0 else None

        #print('(%d/%d) modes done' % (k, self.n_mod)) if k % 3 == 0 else None

        print('save params to file')
        #np.savez(fln, A=self.A, B=self.B)

    def calc_der(self):

        k_x = fftfreq(self.phi_roll.shape[3])*self.phi_roll.shape[3] * 1j
        k_y = fftfreq(self.phi_roll.shape[2])*self.phi_roll.shape[2] * 1j
        self.k = [k_x.reshape(1,k_x.shape[0]), k_y.reshape(k_y.shape[0],1)]

        print('Allocate Space for Arrays')
        self.phi_roll_x = np.zeros((self.phi_roll.shape[0], self.phi_roll.shape[1], self.phi_roll.shape[2], self.phi_roll.shape[3], 2))

        print('Start derivative calculation')

        for var in range(2):
            for mode in range(self.phi_roll.shape[1]):

                temp = fft2(self.phi_roll[var, mode, :, :])

                for dir1 in range(2):
                    self.phi_roll_x[var,mode,:,:,dir1] = np.real(ifft2(self.k[dir1] * temp))

                print('(%d/%d) modes done' % (mode, self.n_mod)) if mode % 5 == 0 and mode != 0 else None
            print('(%d/%d) variables done' % (var+1, 2))


    def calc_der_fd(self):

        print('Allocate Space for Arrays')
        self.phi_roll_x = np.zeros((self.phi_roll.shape[0], self.phi_roll.shape[1], self.phi_roll.shape[2], self.phi_roll.shape[3], 2))

        print('Start derivative calculation')

        for var in range(2):
            for mode in range(self.phi_roll.shape[1]):

                for dir1 in range(2):
                    self.phi_roll_x[var,mode,:,:,dir1] = self.iter_diff(self.phi_roll[var,mode,:,:],dir1)


                print('(%d/%d) modes done' % (mode, self.n_mod)) if mode % 5 == 0 and mode != 0 else None
            print('(%d/%d) variables done' % (var+1, 2))



    def iter_diff(self,data,dir):

        ddx = np.zeros(data.shape)

        if dir == 0:
            ddx[:,0] = (data[:,1] - data[:,-1])/(2*self.dx)
            ddx[:,-1] =(data[:,0] - data[:,-2])/(2*self.dx)

            for i in range(1,data.shape[1]-1):
                ddx[:,i] = (data[:,i+1] - data[:,i-1])/(2*self.dx)

        if dir == 1:
            ddx[0, :] = (data[1, :] - data[-1, :]) / (2 * self.dx)
            ddx[-1, :] = (data[0, :] - data[-2, :]) / (2 * self.dx)

            for j in range(1, data.shape[0] - 1):
                ddx[j, :] = (data[j + 1, :] - data[j - 1, :]) / (2 * self.dx)

        return ddx

#%%

# Functions for autoencoder part

@tf.function #this creates the tf graph
def model_AE(inputs, enc_mods, dec_mods, is_train=False):
    
    '''
    Multiscale autoencoder, taken from Hasegawa 2020. The contribution of the CNNs at different
    scales are simply summed.
    '''
        
    # sum of the contributions of the different CNNs
    encoded = 0
    for enc_mod in enc_mods:
        encoded += enc_mod(inputs, training=is_train)
            
    decoded = 0
    for dec_mod in dec_mods:
        decoded += dec_mod(encoded, training=is_train)
        
    return encoded, decoded

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

def X_to_U_dec(X_flattend, length): # Only works for Nx = Ny
    
    U_2D = np.zeros((X_flattend.shape[0], length, length, 2))
    
    Xu = X_flattend[:, :length*length]
    Xv = X_flattend[:, length*length:]
    
    for j in range(X_flattend.shape[0]):
        for i in range(length):
            U_2D[j, i, : , 0] = Xu[j, length*i : length*i + length]
            U_2D[j, i, : , 1] = Xv[j, length*i : length*i + length]

    return U_2D

def U_to_X_dec(U_input, length):
    
    X_flattened = np.zeros((U_input.shape[0], 2*length*length))
    
    for t in range(U_input.shape[0]):
     for j in range(U_input.shape[1]):
         for i in range(U_input.shape[2]):
             X_flattened[t,i+j*U_input.shape[2]]=U_input[t,j,i,0]
             X_flattened[t,U_input.shape[1]*U_input.shape[2]+i+j*U_input.shape[2]]=U_input[t,j,i,1]
     
    return X_flattened

def X_to_U_enc(X_flattend, N_1): # Only works for Nx = Ny
    
    U_2D = np.zeros((X_flattend.shape[0], N_1[1], N_1[2], N_1[3]))  
    
    for k in range(N_1[3]):
        
        Xu = X_flattend[:, k*N_1[1]*N_1[2] : k*N_1[1]*N_1[2] + N_1[1]*N_1[2] ]
        
        for j in range(X_flattend.shape[0]):
            for i in range(N_1[2]):
                U_2D[j, i, : , k] = Xu[j, N_1[1]*i : N_1[1]*i + N_1[1]]

    return U_2D


def U_to_X_enc(U_input, N_1):
    
    X_flattened = np.zeros((U_input.shape[0], N_1[3]*N_1[2]*N_1[1]))
    
    for k in range(N_1[3]):
        for t in range(U_input.shape[0]):
          for j in range(U_input.shape[1]):
              for i in range(U_input.shape[2]):
                  X_flattened[t, i+j*U_input.shape[2] + U_input.shape[1]*U_input.shape[2]*k]=U_input[t,j,i,k]
         
    return X_flattened

#%%

# Function and class definition

# Hyperparameter Optimization using Grid Search plus Bayesian Optimization

def g(val):
    
    # Gaussian Process reconstruction
    b_e = GPR(kernel = kernell,
            normalize_y = True, # If true mean assumed to be equal to the average of the obj function data, otherwise =0
            n_restarts_optimizer = 3,  # Number of random starts to find the gaussian process hyperparameters
            noise = 1e-10, # Only for numerical stability
            random_state = 1) # Seed    
    
    #Bayesian Optimization
    res = skopt.gp_minimize(val,                         # The function to minimize
                      search_space,                      # The bounds on each dimension of x
                      base_estimator       = b_e,        # GP kernel
                      acq_func             = "EI",       # The acquisition function
                      n_calls              = n_tot,      # Total number of evaluations of f
                      x0                   = x1,         # Initial grid search points to be evaluated at
                      n_initial_points     = n_in,       # The number of additional random initialization points
                      n_restarts_optimizer = 3,          # Number of tries for each acquisition
                      random_state         = 1,          # Seed
                           )   
    return res


def SSV_Noise(x):
    # Recycle Validation
    
    global best_opt, m, enc, dec

    
    # Setting and initializing
    rho      = np.round(x[0], 2)
    input_scaling = np.round(x[1], 2)
    leaking_rate = np.round(x[2],2)
    mean_per_beta = np.zeros(len(beta_array))
    
    for k in range(len(beta_array)):
        
        K.clear_session() # Start a new tensorflow session
        rng = np.random.RandomState(1) # Seed
        beta = beta_array[k] # Seed
        
        model = EchoStateNet.ESN_Hybrid(n_neurons, input_train.shape[2], leaking_rate, rho, sparseness, input_scaling, rng, 'tanh', beta)
        print(beta)
        
        # Train model
        model.fit(input_train, output_train, washout)
        model.reset_states()
        
        horizon = []
        
        for i in range(amount_of_runs): # Loop over 20 runs for the statistics
    
            idx_start = int(i*IC_timeinterval/dt) + resync_idx # IC of this run
            print("idx_start = ", idx_start)
    
            # Initilize the ESN
            x_init = input_all[idx_start - resync_idx: idx_start, :]
            
            # Local reference solution from idx_start till end of integration time
            X_ref_loc = X_ref_test[idx_start:idx_start+len(tt), :]

            # Initialize reservoir
            x_init = x_init.reshape((1, x_init.shape[0], x_init.shape[1]))
            model.predict(x_init)
            
            # Empty list to fill with solutions
            X_pred_enc_list = [] # List of 1D ESN predictions
            X_dec_pred = []
        
            # Take the first value of the local solution as start point for the ESN
            X_last = input_all[idx_start, :] # Start value for time integration
            X_last = X_last.reshape((1, 1, X_last.shape[0])) # Reshape
            
            # Produce temporal predictions with the ESN
        
            #print('start natural response')
            for j in range(tt.shape[0] - 1):
                
                # Predict next time step based on the previous time step X_last
                X_pred_enc_list.append(model.predict(X_last)) # Perform prediction using X_last and store in list
                X_pred_enc = np.array(X_pred_enc_list)[-1] # Take the last value of the list for the next time step                
                
                # Denormalize the predicted encode data
                if normalization == True:
                    X_pred_enc = X_pred_enc * std_o + avg_o
                    
                                    
                # Convert the 2D encode prediction into 2D format
                U_enc_pred_int = X_to_U_enc(X_pred_enc, N_1)
        
                # Decode the predicted data and flatten it into 1D format
                U_dec_pred_int = np.zeros((X_pred_enc.shape[0], len(coord), len(coord), 2))
                U_dec_pred_int[:, :, :, :] = decoder(U_enc_pred_int, dec)
                X_dec_pred_int = U_to_X_dec(U_dec_pred_int, len(coord))
                X_dec_pred.append(X_dec_pred_int)
                
                # compute POD coefficients
                a_last = np.matmul(X_dec_pred_int, phi_red)
                
                # integrate last coefficients
                a_last = podgl.integrate(a_last[0, :])
                
                # PODGL state at next time step
                X_podgl_new_dec = np.matmul(a_last, np.transpose(phi_red))
                X_podgl_new_dec = np.reshape(X_podgl_new_dec, (1, X_podgl_new_dec.shape[0]))
                
                # Encode the PODGL solution again
                
                # First write PODGL solution in 2D format       
                U_podgl_new_dec = X_to_U_dec(X_podgl_new_dec, len(coord))
                
                # Encode the PODGL solution 
                U_podgl_new_enc = np.zeros((1, N_1[1], N_1[2], N_1[3]))
                U_podgl_new_enc[:,:,:,:] = model_AE(U_podgl_new_dec, enc, dec)[0]
                
                # Return the encoded solution back into the 1D format
                X_podgl_new_enc = U_to_X_enc(U_podgl_new_enc, N_1)
                
                # Form new input vector
                X_last = np.concatenate((X_podgl_new_enc, X_pred_enc), axis=1)
                
                # Normalize again

                if normalization == True:
                    X_last = (X_last - avg_i) / std_i
                
                # Reshape x_last
                X_last = X_last.reshape(1, 1, X_last.shape[1])
                                
            X_dec_pred = np.array(X_dec_pred)
            X_dec_pred = X_dec_pred.reshape(X_dec_pred.shape[0], X_dec_pred.shape[2])
            
            # Add first timestep to array
            X_dec_pred = np.vstack((X_ref_loc[0, :], X_dec_pred))
            
            model.reset_states()       
            
            # Calculate Error
            err = LA.norm(X_dec_pred - X_ref_loc, axis=1) / np.sqrt(
                np.average(np.square(LA.norm(X_ref_loc, axis=1))))                       
                            
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
epsilon = 0.4 # Threshold of the allowed error
skip_dim = 8622 # Number of truncated modes
plot = True # Make the plot
modesretained = 8712 - skip_dim # Number of retained modes
normalization = True

file =  np.load("../../../Folder_of_AE/AE_3C_3L_e12_24_5_d24_12_6_N8_8_5/Encoded_data_Re34.npz")
path = '../../../Folder_of_AE/AE_3C_3L_e12_24_5_d24_12_6_N8_8_5/Kflow_RE34_AE' # To extract autoencder

Re = file["Re"] # Reynolds number
Lyap = 0.067 # Lyapunov exponent
dt = file["dt"] # Dt between data samples
coord = file["coord"]
X_enc = file["U_enc"] # Encoded data
n_epochs = file["n_epochs"] # Number of epochs in autoencoder traning
N_parallel = file["N_parallel"] # Number of parallel encoders/decoders
ker_size = file["ker_size"] # Kernel/filter sizes used
N_latent = file["N_latent"] # Total latent space dimensions
N_1 = file["N_1"] # Latent space dimensions

X_enc = X_enc[:90000]

# Read reference data
file_ref = np.load("../../../Reduced_Kolmogorov_Re34.npz")
X_ref = file_ref["X"]

gridpoints = len(coord)
nk = int((gridpoints - 1)/2)

#%%

#ESN setup
K.clear_session()
rng = np.random.RandomState(1)

n_neurons = 500 # Number of neurons in reservoir
degree = 3.5  # Average number of connections of a unit to other units in reservoir
sparseness = 1. - degree / (n_neurons - 1.)  # sparseness of W

# Training and validation settings
upsample = 1 # Increase delta t by a factor of upsample to decrease computational time
dt = dt*upsample
X_enc = X_enc[::upsample] # Skip every upsample-th data
X_ref = X_ref[::upsample]

data_len = X_enc.shape[0]

T_lyap = 1/Lyap # Lyapunov time
washout   = 100 # Washout
t_train   = 500*T_lyap # Training time
N_train   = int(t_train/dt) # Train certain number of lyapunov times
t_stop    = int(20*T_lyap) # Predict certain number of lyapunov times
tt = np.arange(0,t_stop+dt,dt)/T_lyap # Time array, per measurement

# Resync time for the ESN to burn in
resync_time = 20*T_lyap
resync_idx = int(resync_time/dt)

amount_of_runs = 2 # Number of runs in per hyperparameter setting
potential_time_lastIC = t_train - t_stop - resync_time # Last step of the training time minus 1 prediction set
IC_timeinterval = potential_time_lastIC/amount_of_runs  # Amount of timestep between IC_snapshots

#%%

# Set up the PODGL solver

pod = POD()
eig, a, phi = pod(X_ref,False)

# Reduce dimension
a_red = a[:,0:-skip_dim]
phi_red = phi[:,0:-skip_dim]

# Generate ROM
fln = './A_B_' + str(skip_dim) + '.npz'
podgl = ROM(phi_red,coord,dt,Re,fln)

# Create array with integrated ROM coefficients

a_podgl = np.zeros((a.shape[0],a_red.shape[1]))
for i in range(a.shape[0]):

    a_podgl[i,:] = podgl.integrate(a_red[i,:])

    print('(%d/%d) steps done' % (i, a.shape[0])) if i % 50 == 0 else None

# Transform to physical space
X_podgl = np.matmul(a_podgl, np.transpose(phi_red))


#%%

# Encode PODGL solution for the trainingsdata

# Turn PODGL X solution into 2D form
U_flow_podgl = X_to_U_dec(X_podgl, gridpoints)   
 
# Encode the POD GL solution    

# Split in k interval of N_pos length needed to process long timeseries
N_pos     = 23
count     = int(data_len/N_pos)

N_x       = U_flow_podgl.shape[1]
N_y       = U_flow_podgl.shape[2] 
enc = [None]*(int(N_parallel))
dec = [None]*(int(N_parallel))

# Load the encoder and decoder models
for i in range(N_parallel):
    enc[i] = tf.keras.models.load_model(path + '/enc_mod'+str(tuple(ker_size[i]))+'_'+str(N_latent)+'.h5', 
                                          custom_objects={"PerPad2D": PerPad2D})
for i in range(N_parallel):
    dec[i] = tf.keras.models.load_model(path + '/dec_mod'+str(tuple(ker_size[i]))+'_'+str(N_latent)+'.h5',
                                          custom_objects={"PerPad2D": PerPad2D})

U_enc_podgl = np.zeros((data_len, N_1[1], N_1[2], N_1[3]))

# Encode all the data to provide time series in latent space for the ESN
for i in range(count):
    U_enc_podgl[i*N_pos:(i+1)*N_pos]= model_AE(U_flow_podgl[i*N_pos:(i+1)*N_pos], enc, dec)[0]

U_enc_podgl[count*N_pos:]= model_AE(U_flow_podgl[count*N_pos:], enc, dec)[0]

# Now flatten the encoded data into 1D format
X_enc_podgl = U_to_X_enc(U_enc_podgl, N_1)

#%%

# Concatenate PODGL and real solution
input_all = np.concatenate((X_enc_podgl, X_enc), axis=1)

# Create mapping
input_all = input_all[:-1, :]
output_all = X_enc[1:, :]

# Normalize for better prediction
if normalization == True:
    avg_i = np.average(input_all,axis=0)
    std_i = np.std(input_all, axis=0)
    avg_o = np.average(output_all,axis=0)
    std_o = np.std(output_all, axis=0)

    input_all = (input_all - avg_i) / std_i
    output_all = (output_all - avg_o) / std_o

input_train  = input_all[:N_train] #inputs
output_train  = output_all[:N_train] #data to match at next timestep

input_train = input_train.reshape((1,input_train.shape[0],input_train.shape[1]))

# Reference solution for prediction time interval
X_ref_test = X_ref[:, :]
X_enc_test = X_enc[:, :]

#%%

# Baysian optimisation and grid search preparation

# Range for hyperparameters
rho_in     = 0.01    
rho_end    = 5.0 
input_scaling_in  = 0.01
input_scaling_end = 5.0
leaking_rate_in = 0.1
leaking_rate_end = 1.0

n_in  = 7        # Number of Initial random points

# In case we want to start from a grid_search, the first n_grid_x*n_grid_y points are from grid search
n_grid_x = 0 # if zero, the grid search is turned off
n_grid_y = n_grid_x
n_grid_z = n_grid_x
n_bo     = 7  # Number of points to be acquired through BO after grid search
n_tot    = n_grid_x*n_grid_y*n_grid_z + n_bo + n_in #Total Number of Function Evaluatuions

# Computing the points in the grid
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
                   
  
# Range for hyperparameters, can be more than 2 but note that also the initial grid should be adapted accordingly!
search_space = [Real(rho_in, rho_end, name='spectral_radius'),
                Real(input_scaling_in, input_scaling_end, name='input_scaling'),
                Real(leaking_rate_in, leaking_rate_end, name='leaking_rate')]

# ARD 5/2 Matern Kernel with sigma_f in front for the Gaussian Process
kernell = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-1, 3e0))*\
                  Matern(length_scale=[0.2,0.2, 0.1], nu=2.5, length_scale_bounds=(5e-2, 1e1))  
                  
#%%

# Hyperparameter search set-up

# Number of Networks in the ensemble
ensemble = 1

# Which validation strategy (implemented in Val_Functions.ipynb)
val      = SSV_Noise                 # validation method

# Quantities to be saved
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
























# %%
