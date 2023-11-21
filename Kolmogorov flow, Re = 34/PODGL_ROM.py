"""
Knowledge-based model architecture for predictions on high dimensional systems by Jochem Veerman 
for his Msc thesis project on hybrid knowledge-based/deep learning reduced order modelling 
for high dimensional chaotic systen. The code is obtained and slightly adapted from:

Lesjak and Doan, 2021: 
Chaotic systems learning with hybrid echo state network/proper orthogonal decomposition based model
"""


#%%

from os import environ
environ['OMP_NUM_THREADS'] = "1"

import os.path
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy.linalg as LA
from scipy.fft import fft2, ifft2, fftfreq, fft, ifft
from matplotlib import cm
import matplotlib.animation as animation
#import POD
import time

start = time.time()

#%%

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
        eig, phi = np.linalg.eigh(C)

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


        self.N = int((coord.shape[0]-1)/2) # Amount of modes in spectral space
        self.n_mod = phi.shape[1] # Amount of reduced modes obtained by truncation of POD
        n = 4 # Frequency of the sinusoidal Kolmogorov foring
        self.Re = Re # Reynolds number

        # physical grid
        self.dx = coord[1]-coord[0] # 1 spatial step in physical grid
        xx, yy = np.meshgrid(coord, coord) # Create grid
        fx = np.sin(n * yy) # Compute Kolmogorov forcing based on grid 

        self.phi = phi # The reduced set of modes
        
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
        self.h = h # dt = small step in time
        w, v = LA.eig(self.A) # Compute eigenvalues and eigenvectors of linear operator
        inv_v = LA.inv(v) # Take the inverse of the eigenvectors

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

        return temp + self.fx_trfd # Include the projected forcing term

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

                    #print('(%d/%d) in total' % ((k*self.n_mod*self.n_mod)+(m*self.n_mod)+n, self.n_mod**3)) if ((k*self.n_mod*self.n_mod)+(m*self.n_mod)+n) % 100000 == 0 else None

        #print('(%d/%d) modes done' % (k, self.n_mod)) if k % 3 == 0 else None

        print('save params to file')
        #np.savez(fln, A=self.A, B=self.B)

    def calc_der(self):        
        
        k_x = fftfreq(self.phi_roll.shape[3])*self.phi_roll.shape[3] * 1j
        k_y = fftfreq(self.phi_roll.shape[2])*self.phi_roll.shape[2] * 1j
        self.k = [k_x.reshape(1,k_x.shape[0]), k_y.reshape(k_y.shape[0],1)]


        # HEre the last index gives the dimensions in which the derivative is taken as can be seen from the next lines.
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

def X_to_U_dec(X_flattend, length): # Only works for Nx = Ny
    
    U_2D = np.zeros((X_flattend.shape[0], length, length, 2))
    
    Xu = X_flattend[:, :length*length]
    Xv = X_flattend[:, length*length:]
    
    for j in range(X_flattend.shape[0]):
        for i in range(length):
            U_2D[j, i, : , 0] = Xu[j, length*i : length*i + length]
            U_2D[j, i, : , 1] = Xv[j, length*i : length*i + length]

    return U_2D

#%%

#setup
epsilon = 0.4 # Threshold of the allowed error
skip_dim = 8712-90 # Number of modes to truncation
modesretained = 8712 - skip_dim # Number of retained modes

#%%

# Read data from files
np_file = np.load("Reduced_Kolmogorov_Re34.npz") # Load reference data
X = np_file['X']  # Data
dt = np_file['dt'] # Sampling time step of data
Lyap = 0.067 # Lyapunov exponent
coord = np_file['coord']
Re = np_file['Re'] # Reynolds number
dt_solver = np_file["dt_solver"]
tmax = np_file["t_max"]

gridpoints = len(coord)
nk = int((gridpoints - 1)/2)

#%%

# Alter time step with a factor upsample
upsample = 1 
dt = dt*upsample 
X = X[::upsample]

T_lyap = 1/Lyap # Lyapunov time
t_train   = 500*T_lyap # Number of training time steps
N_train   = int(t_train/dt) # Train certain number of Lyapunov times
t_stop_LT = int(5*T_lyap) # Prediction time
tt = np.arange(0,t_stop_LT+dt,dt)/T_lyap # time array, per measurement

#%%

# Decomposition
pod = POD()
eig, a, phi = pod(X,False) # Due to the False, we do not substract the time average velocity and this is thus included in the eigenmodes

#print(eig)
#print(eig/np.sum(eig))
#plt.plot(eig/np.sum(eig))
#plt.show()

# Reduce dimension
a_red = a[:,0:-skip_dim]
phi_red = phi[:,0:-skip_dim]

# Generate ROM
fln = './A_B_' + str(skip_dim) + '.npz'
rom = ROM(phi_red,coord,dt,Re,fln)

#%%

idx_start = 76715 # IC of this run, 

# mMin loop
a_last = np.zeros((tt.shape[0],a_red.shape[1]))
a_last[0,:] = a_red[idx_start,:]

for i in range(len(tt)-1):

    a_last[i+1,:] = rom.integrate(a_last[i,:])
    print('(%d/%d) steps done' % (i, tt.shape[0]-1)) if i % 50 == 0 else None

#%%

x_pred = np.matmul(a_last,np.transpose(phi_red)) # Transform prediction solution to physical space
x_loc = X[idx_start:x_pred.shape[0] + idx_start,:] # Reference solution at same time steps

# calculate Error
err = LA.norm(x_pred - x_loc, axis=1) / np.sqrt(
    np.average(np.square(LA.norm(x_loc, axis=1))))
print((np.argmax(err > epsilon) + 1) * dt/T_lyap)

# Calculate Error
plt.plot(tt,err)
plt.axvline((np.argmax(err > epsilon) + 1) * dt/T_lyap,0,1,label='valid time',color='red')
plt.xlabel("t")
plt.ylabel("RMSE")
plt.show()