"""
Direct numerical simulations of two-dimensional Kolmogorov flow using the 
Kolsol library in Numpy (Python).
"""



import numpy as np
import matplotlib.pyplot as plt
from kolsol.numpy.solver import KolSol
import matplotlib.animation as animation
import time

start = time.time()

#%%

def RK4(q0,dt,N,func):
    ''' 4th order RK for autonomous systems described by func '''

    for i in range(N):

        k1   = dt * func(q0)
        k2   = dt * func(q0 + k1/2)
        k3   = dt * func(q0 + k2/2)
        k4   = dt * func(q0 + k3)
        q0   = q0 + (k1 + 2*k2 + 2*k3 + k4)/6

    return  q0

#%%

# Solver settings
Re = 34 # Reynolds number
Nk = 32 # Number of Fourier modes
Nf = 4 # Forcing frequency
Phys_size1 = 2*Nk + 2 # Grid size in physical space, must be even number
fouriersize = 2*Nk+1 # Grid size in Fourier space
dt = 0.01 # Time step
dt_solver = 0.01
t_max = 100000 # Maximum integration time 
N = int(t_max/dt) # Total number of integration steps 
t_trans = 1000 # Transient time
N_trans = int(t_trans/dt) # Number of transient step
x = np.linspace(0, 2*np.pi, Phys_size1+1) # Physical data
x = x[:-1]
xx, yy = np.meshgrid(x,x)

# Counters to aid for easy store of data
count = 0
count_storage = 0

upsample = 0.1 # Time step for data storage
 
time_array = np.arange(0.0, t_max, dt) # amount of time units
time_array = np.round(time_array, 2)
u_data = np.zeros([Phys_size1, Phys_size1, int(t_max/upsample)])
v_data = np.zeros([Phys_size1, Phys_size1, int(t_max/upsample)])

#%%

ks = KolSol(nk=Nk, nf=Nf, re=Re, ndim=2) # Set up Kolsol library
u_hat = np.ones([fouriersize, fouriersize, 2], dtype = complex) # fixed initial condition

# Loop over the transient using RK4
for j in range(N_trans):
    k1   = dt * ks.dynamics(u_hat)
    k2   = dt * ks.dynamics(u_hat + k1/2)
    k3   = dt * ks.dynamics(u_hat + k2/2)
    k4   = dt * ks.dynamics(u_hat + k3)

    u_hat   = u_hat + (k1 + 2*k2 + 2*k3 + k4)/6

print("Done with transient calculation") # Transient computations done

# Loop over the total number of points for the main integration using RK4
for i in range(N):
    k1   = dt * ks.dynamics(u_hat)
    k2   = dt * ks.dynamics(u_hat + k1/2)
    k3   = dt * ks.dynamics(u_hat + k2/2)
    k4   = dt * ks.dynamics(u_hat + k3)

    u_hat   = u_hat + (k1 + 2*k2 + 2*k3 + k4)/6
    
    # Move to physical space
    u_field = ks.fourier_to_phys(u_hat, nref = Phys_size1) # Move to physical space
    
    # Store the data
    
    if count%(upsample/dt) == 0: # Save data at a certain sampling rate.
            u_data[:,:,count_storage] = np.transpose(u_field[:,:,0])
            v_data[:,:,count_storage] = np.transpose(u_field[:,:,1])
            count_storage += 1     
        
    count += 1 
       
print("Done with all computations")

U = np.zeros((u_data.shape[2],2*u_data.shape[0]*u_data.shape[1]))

print('Flatten first array')
for t in range(u_data.shape[2]):
    for j in range(u_data.shape[0]):
        for i in range(u_data.shape[1]):
            U[t,i+j*u_data.shape[1]]=u_data[j,i,t]
            U[t,u_data.shape[0]*u_data.shape[1]+i+j*u_data.shape[1]]=v_data[j,i,t]
            

end = time.time()

#Subtract Start Time from The End Time
total_time = end - start
print("\n"+ str(total_time))

print('Save file')
fln = './Kolmogorov_Re' + str(Re) + " Nk = " +  str(Nk)
np.savez(fln, X=U, coord=x, Re=Re, dt=upsample, dt_solver=dt_solver, t_max=t_max, t_trans=t_trans, Nk=Nk, Nf=Nf)