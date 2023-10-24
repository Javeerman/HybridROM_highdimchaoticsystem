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
import importlib.util
spec1 = importlib.util.spec_from_file_location("POD", "../POD.py") #Import POD module
propdec = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(propdec)

start = time.time()

#%%

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

#setup
epsilon = 0.4 # Threshold of the allowed error
skip_dim = 648 - 18 # How many dimensions to skip
plot = True # Make the plot
modesretained = 648 - skip_dim

#%%

# Read data from files
np_file = np.load("Kolmogorov_Re20.npz") #np.load("Kolmogorov_Re20.0_T2000_DT01.npz")
X = np_file['X']  # Data
dt = np_file['dt']
Lyap = np_file['Lyap']
coord = np_file['coord']
Re = np_file['Re']

gridpoints = len(coord)
nk = int((gridpoints - 1)/2)

#%%

upsample = 5
dt = dt*upsample # if we gonna skip every upsample, dt becomes a factor upsample bigger
X = X[::upsample]

T_lyap = 1/Lyap
N_lyap = int(T_lyap/dt)
washout   = 100
N_train   = 500*N_lyap
t_stop    = 1000
tt = np.arange(0,t_stop+dt,dt)/T_lyap # time array, per measurement


#%%

"""Here we are going to compute the POD modes, which will return a matrix of size
(2*Nx*NY)X(2*Nx*nY)"""

#Decomposition
pod = propdec.POD()
eig, a, phi = pod(X,False) # Due to the False, we do not substract the time average velocity and this is thus included in the eigenmodes

#print(eig)
#print(eig/np.sum(eig))
#plt.plot(eig/np.sum(eig))
#plt.show()

#reduce dimension
a_red = a[:,0:-skip_dim]
phi_red = phi[:,0:-skip_dim]

#generate ROM
fln = './A_B_' + str(skip_dim) + '.npz'
rom = ROM(phi_red,coord,dt,Re,fln)

#%%

# Parameters for time integration
idx_start = 20000 + int(N_train) # IC of this run, 

# main loop
a_last = np.zeros((tt.shape[0],a_red.shape[1]))
a_last[0,:] = a_red[idx_start,:]

x_loc = X[idx_start:idx_start+len(tt), :]

for i in range(len(tt)-1):

    a_last[i+1,:] = rom.integrate(a_last[i,:])
    print('(%d/%d) steps done' % (i, tt.shape[0]-1)) if i % 50 == 0 else None

#%%

x_pred = np.matmul(a_last,np.transpose(phi_red))

#%%

err = LA.norm(x_pred - x_loc, axis=1) / np.sqrt(
    np.average(np.square(LA.norm(x_loc, axis=1))))

if plot == True:

    # calculate Error
    t_horizon_plot = 1000
    plt.plot(tt,err)
    plt.axvline((np.argmax(err > epsilon) + 1) * dt/T_lyap,0,1,label='valid time',color='red')
    plt.xlabel("t/tL")
    plt.ylabel("RMSE")
    plt.xlim(0, t_horizon_plot)
    plt.title("Prediction horizon PODGL, retained modes = %s, idx = %s" %(modesretained, idx_start))
    plt.show()

sum_eig = np.sum(eig)
norm_eig = eig/sum_eig


XXXX = np.arange(0, len(norm_eig), 1)
plt.figure()
plt.plot(XXXX[::5], norm_eig[::5], 'k.')
plt.yscale("log")
plt.ylabel(r"$\dfrac{\lambda_i}{\sum \lambda_i}$", size = 16)
plt.xlabel(r"POD-modes $\Phi_j$ ", size = 16)
plt.xticks(size = 14)
plt.yticks(size = 14)
plt.grid()
plt.xlim(0,500)
plt.tight_layout()
plt.savefig("POD_energy_Re20.pdf", dpi = 200)
plt.show()

#%%
print('It took', time.time()-start, 'seconds.')

# COMPUTE LONG TERM STATISTICS OF THE REFERENCE SOLUTION

# Compute kinetic energy and dissipaition as function of time
Ekin_ref = np.zeros(len(tt))
Dissap_ref = np.zeros(len(tt))
u2D_flow_gradx_ref = np.zeros((len(tt), gridpoints, gridpoints))
v2D_flow_grady_ref =   np.zeros((len(tt), gridpoints, gridpoints))

# split uu_matrix into u and v matrix
u1D_flow_ref = x_loc[:, :gridpoints**2]
v1D_flow_ref = x_loc[:, gridpoints**2:]

u2D_flow_ref = np.zeros([u1D_flow_ref.shape[0], gridpoints, gridpoints])
v2D_flow_ref = np.zeros([u1D_flow_ref.shape[0], gridpoints, gridpoints])

for j in range(len(tt)):
    for i in range(gridpoints):
        u2D_flow_ref[j, i, :] = u1D_flow_ref[j, gridpoints*i: gridpoints*i + gridpoints]
        v2D_flow_ref[j, i, :] = v1D_flow_ref[j, gridpoints*i: gridpoints*i + gridpoints]
   
# Compute Ekin and dissipation
for i in range(len(tt)):
        half_norm_squared_ekin = 0.5*(u2D_flow_ref[i, :, :]**2 + v2D_flow_ref[i, :, :]**2)
        Ekin_ref[i] = 1/(2*np.pi)**2 * np.sum(half_norm_squared_ekin*((2*np.pi)/(gridpoints))**2)

        u2D_flow_gradx_ref[i,:,:] = np.gradient(u2D_flow_ref[i,:,:], 2*np.pi/gridpoints)[1]
        v2D_flow_grady_ref[i,:,:] = -np.gradient(v2D_flow_ref[i,:,:], 2*np.pi/gridpoints)[0]
        
        local_dissap_ref = 1/Re*(np.sqrt(u2D_flow_gradx_ref[i,:,:]**2 + v2D_flow_grady_ref[i,:,:]**2))**2
        
        Dissap_ref[i] = 1/(2*np.pi)**2 * np.sum((local_dissap_ref)*((2*np.pi)/(gridpoints))**2)
        
plt.figure()
plt.plot(Ekin_ref,Dissap_ref, "k.", label = "D(t) vs Ekin(t)")
plt.ylabel("D(t)")
plt.xlabel("Ekin(t)")
plt.plot(Ekin_ref[0], Dissap_ref[0], "ro", markersize = 16, label = "t = 0")
plt.legend()
plt.title("Dissipation vs Ekin of reference solution")
plt.show()

fig, ax = plt.subplots(figsize=(11,7))
plt.plot(tt,Ekin_ref, "k")
plt.grid()
plt.xlabel(r"$t$", size = 26)
plt.ylabel(r"$E(t)$", size = 26)
ax.tick_params(axis='both', which='major', labelsize=22)
plt.tight_layout()
plt.savefig(r"Ekin_vs_t_Kolmogorov_Re20.pdf", dpi = 300)
plt.show()

# Compute the time averaged flow

u2D_flow_mean_ref = np.mean(u2D_flow_ref, axis = 0)
v2D_flow_mean_ref = np.mean(v2D_flow_ref, axis = 0)

plt.figure()
plt.imshow(u2D_flow_mean_ref, origin = "lower", extent = [0, 2*np.pi, 0, 2*np.pi])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("time average of u, reference")
plt.show()

plt.figure()
plt.imshow(v2D_flow_mean_ref, origin = "lower", extent = [0, 2*np.pi, 0, 2*np.pi])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("time average of v, reference")
plt.show()

# Compute the time averaged of the fluctuations

u2D_flow_flucmean_ref = np.sqrt((np.mean(u2D_flow_ref - u2D_flow_mean_ref, axis = 0))**2)
v2D_flow_flucmean_ref = np.sqrt((np.mean(v2D_flow_ref - v2D_flow_mean_ref, axis = 0))**2)

plt.figure()
plt.imshow(u2D_flow_flucmean_ref, origin = "lower", extent = [0, 2*np.pi, 0, 2*np.pi])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("time average of fluc u, reference")
plt.show()

plt.figure()
plt.imshow(v2D_flow_flucmean_ref, origin = "lower", extent = [0, 2*np.pi, 0, 2*np.pi])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("time average of fluc v, reference")
plt.show()

# #%%

# # COMPUTE LONG TERM STATISTICS OF THE PREDICITIONS

# Compute kinetic energy and dissipaition as function of time
Ekin_ROM = np.zeros(len(tt))
Dissap_ROM = np.zeros(len(tt))
u2D_flow_gradx_ROM = np.zeros((len(tt), (gridpoints), (gridpoints)))
v2D_flow_grady_ROM = np.zeros((len(tt), (gridpoints), (gridpoints)))

# split uu_matrix into u and v matrix
u1D_flow_ROM = x_pred[:, :gridpoints**2]
v1D_flow_ROM = x_pred[:, gridpoints**2:]

u2D_flow_ROM = np.zeros([u1D_flow_ROM.shape[0], (gridpoints), (gridpoints)])
v2D_flow_ROM = np.zeros([u1D_flow_ROM.shape[0], (gridpoints), (gridpoints)])

for j in range(len(tt)):
    for i in range(gridpoints):
        u2D_flow_ROM[j, i, :] = u1D_flow_ROM[j, (gridpoints)*i: (gridpoints)*i + (gridpoints)]
        v2D_flow_ROM[j, i, :] = v1D_flow_ROM[j, (gridpoints)*i: (gridpoints)*i + (gridpoints)]
   
# # Compute Ekin and dissipation
for i in range(len(tt)):
        half_norm_squared_ekin_ROM = 0.5*(u1D_flow_ROM[i, :]**2 + v1D_flow_ROM[i, :]**2)
        Ekin_ROM[i] = 1/(2*np.pi)**2 * np.sum((half_norm_squared_ekin_ROM)*((2*np.pi)/(gridpoints))**2)
        
        u2D_flow_gradx_ROM[i,:,:] = np.gradient(u2D_flow_ROM[i,:,:], 2*np.pi/gridpoints)[1]
        v2D_flow_grady_ROM[i,:,:] = -np.gradient(v2D_flow_ROM[i,:,:], 2*np.pi/gridpoints)[0]
        
        local_dissap_ROM = 1/Re*(np.sqrt(u2D_flow_gradx_ROM[i,:,:]**2 + v2D_flow_grady_ROM[i,:,:]**2))**2
        
        Dissap_ROM[i] = 1/(2*np.pi)**2 * np.sum((local_dissap_ROM)*((2*np.pi)/(gridpoints))**2)
        
plt.figure()
plt.plot(Ekin_ROM,Dissap_ROM, "k.", label = "D(t) vs Ekin(t)")
plt.ylabel("D(t)")
plt.xlabel("Ekin(t)")
plt.plot(Ekin_ROM[0], Dissap_ROM[0], "ro", markersize = 16, label = "t = 0")
plt.legend()
plt.title("Dissipation vs Ekin of PODGL solution, retained modes = %s" %modesretained)
plt.show() 

plt.figure()
plt.plot(tt,Ekin_ROM, "ko", label = "Ekin(t)")
plt.ylabel("t")
plt.xlabel("Ekin(t)")
plt.plot(tt[0], Ekin_ROM[0], "ro", markersize = 16, label = "t = 0")
plt.legend()
plt.title("Ekin vs time of reference solution")
plt.show()

# Compute the time averaged flow

u2D_flow_mean_ROM = np.mean(u2D_flow_ROM, axis = 0)
v2D_flow_mean_ROM = np.mean(v2D_flow_ROM, axis = 0)

plt.figure()
plt.imshow(u2D_flow_mean_ROM, origin = "lower", extent = [0, 2*np.pi, 0, 2*np.pi])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("time average of u, PODGL")
plt.show()

plt.figure()
plt.imshow(v2D_flow_mean_ROM, origin = "lower", extent = [0, 2*np.pi, 0, 2*np.pi])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("time average of v, PODGL")
plt.show()       
        
# Compute the time averaged of the fluctuations

u2D_flow_flucmean_ROM = np.sqrt((np.mean(u2D_flow_ROM - u2D_flow_mean_ROM, axis = 0))  **2)
v2D_flow_flucmean_ROM = np.sqrt((np.mean(v2D_flow_ROM - v2D_flow_mean_ROM, axis = 0))  **2)

plt.figure()
plt.imshow(u2D_flow_flucmean_ROM, origin = "lower", extent = [0, 2*np.pi, 0, 2*np.pi])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("time average of fluc u, PODGL")
plt.show()

plt.figure()
plt.imshow(v2D_flow_flucmean_ROM, origin = "lower", extent = [0, 2*np.pi, 0, 2*np.pi])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.title("time average of fluc v, PODGL")
plt.show()

# #%%

# # #Make animation of U

# # writer = animation.ImageMagickWriter(fps=1000)

# skip= 500
# N_gif=tt.shape[0]-1
# frps=5

# fig1, (ax2) = plt.subplots(1)

# ims_u = []

# for i in range(0,N_gif,skip):
#     #im1 = ax1.imshow(u2D_flow_ROM[i,:,:],interpolation='bicubic',cmap='coolwarm',animated=True, extent=[0,2*np.pi,0,2*np.pi],vmax=1,vmin=-1)
#     #ax1.set_title("u: POD_GL ROM")
#     #ax1.set_xlabel("x")
#     #ax2.set_ylabel("y")
#     im2 = ax2.imshow(u2D_flow_ref[i,:,:],interpolation='bicubic',cmap='coolwarm', origin = "lower",animated=True, extent=[0,2*np.pi,0,2*np.pi],vmax=1,vmin=-1)
#     ax2.set_title("u: Reference DNS", fontsize = 18)
#     ax2.set_xlabel("x", fontsize = 16)
#     ax2.set_ylabel("y", fontsize = 16)
#     plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'], fontsize = 16)
#     plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'], fontsize = 16)
#     ims_u.append([im2])
# plt.colorbar(im2)    

# ani = animation.ArtistAnimation(fig1, ims_u, interval=100, blit=True,
#                                 repeat_delay=1000)
# ani.save('./u_PODGL_rom+ref_retained modes = %s.gif' %modesretained ,fps=frps, writer='imagemagick')

# #%%

# # Make animation of V

# #writer = animation.ImageMagickWriter(fps=1000)

# plt.clf()

# fig2, (ax2) = plt.subplots(1)

# ims_v = []

# for i in range(0,N_gif,skip):
#     #im1 = ax1.imshow(v2D_flow_ROM[i,:,:],interpolation='bicubic',cmap='coolwarm',animated=True, extent=[0,2*np.pi,0,2*np.pi],vmax=1,vmin=-1)
#     #ax1.set_title("v: POD_GL ROM")
#     #ax1.set_xlabel("x")
#     ax2.set_ylabel("y")
#     im2 = ax2.imshow(v2D_flow_ref[i,:,:],interpolation='bicubic',cmap='coolwarm', origin = "lower",animated=True, extent=[0,2*np.pi,0,2*np.pi],vmax=1,vmin=-1)
#     ax2.set_title("v: Reference DNS", fontsize = 18)
#     ax2.set_xlabel("x", fontsize = 16)
#     ax2.set_ylabel("y", fontsize = 16)
#     plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'], fontsize = 16)
#     plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'], fontsize = 16)
#     ims_v.append([im2])
# plt.colorbar(im2)    


# ani = animation.ArtistAnimation(fig2, ims_v, interval=100, blit=True,
#                                 repeat_delay=1000)
# ani.save('./v_PODGL_ROM+ref_retainedmodes = %s.gif' %modesretained,fps=frps, writer='imagemagick')

# plt.clf()

# # Make animation of norm

# #writer = animation.ImageMagickWriter(fps=1000)

# plt.clf()

# fig2, (ax2) = plt.subplots(1)

# ims_v = []

# for i in range(0,N_gif,skip):
#     #im1 = ax1.imshow(v2D_flow_ROM[i,:,:],interpolation='bicubic',cmap='coolwarm',animated=True, extent=[0,2*np.pi,0,2*np.pi],vmax=1,vmin=-1)
#     #ax1.set_title("v: POD_GL ROM")
#     #ax1.set_xlabel("x")
#     ax2.set_ylabel("y")
#     im2 = ax2.imshow(np.sqrt(v2D_flow_ref[i,:,:]**2 + u2D_flow_ref[i,:,:]**2),interpolation='bicubic',cmap='coolwarm', origin = "lower",animated=True, extent=[0,2*np.pi,0,2*np.pi],vmax=1,vmin=-1)
#     ax2.set_title("Reference DNS, Re20", fontsize = 18)
#     ax2.set_xlabel("x", fontsize = 16)
#     ax2.set_ylabel("y", fontsize = 16)
#     plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'], fontsize = 16)
#     plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'], fontsize = 16)
#     ims_v.append([im2])
# plt.colorbar(im2)    


# ani = animation.ArtistAnimation(fig2, ims_v, interval=100, blit=True,
#                                 repeat_delay=1000)
# ani.save('./v_PODGL_ROM+ref_retainedmodes = %s.gif' %modesretained,fps=frps, writer='imagemagick')

# plt.clf()


# # ###############################################################

# # # Plot Modes from POD


# # idx = 1000

# # true_sol = u2D_flow_ref[idx,:,:]
# # a_sol = a[idx, :]
# # phi1 = phi[:,0]
# # phi2 = phi[:,1]
# # phi3 = phi[:,2]

# # # split uu_matrix into u and v matrix
# # u1D_flow_phi1 = phi1[:gridpoints**2]
# # v1D_flow_phi1 = phi1[gridpoints**2:]

# # u2D_flow_phi1 = np.zeros([(gridpoints), (gridpoints)])
# # v2D_flow_phi1 = np.zeros([(gridpoints), (gridpoints)])

# # for i in range(gridpoints):
# #         u2D_flow_phi1[i, :] = u1D_flow_phi1[(gridpoints)*i: (gridpoints)*i + (gridpoints)]
# #         v2D_flow_phi1[i, :] = v1D_flow_phi1[(gridpoints)*i: (gridpoints)*i + (gridpoints)]
        
# # # split uu_matrix into u and v matrix
# # u1D_flow_phi2 = phi2[:gridpoints**2]
# # v1D_flow_phi2 = phi2[gridpoints**2:]

# # u2D_flow_phi2 = np.zeros([(gridpoints), (gridpoints)])
# # v2D_flow_phi2 = np.zeros([(gridpoints), (gridpoints)])

# # for i in range(gridpoints):
# #         u2D_flow_phi2[i, :] = u1D_flow_phi2[(gridpoints)*i: (gridpoints)*i + (gridpoints)]
# #         v2D_flow_phi2[i, :] = v1D_flow_phi2[(gridpoints)*i: (gridpoints)*i + (gridpoints)]
        
# # # split uu_matrix into u and v matrix
# # u1D_flow_phi3 = phi3[:gridpoints**2]
# # v1D_flow_phi3 = phi3[gridpoints**2:]

# # u2D_flow_phi3 = np.zeros([(gridpoints), (gridpoints)])
# # v2D_flow_phi3 = np.zeros([(gridpoints), (gridpoints)])

# # for i in range(gridpoints):
# #         u2D_flow_phi3[i, :] = u1D_flow_phi3[(gridpoints)*i: (gridpoints)*i + (gridpoints)]
# #         v2D_flow_phi3[i, :] = v1D_flow_phi3[(gridpoints)*i: (gridpoints)*i + (gridpoints)]

# # plt.figure()
# # plt.imshow(true_sol,interpolation='bicubic',cmap='coolwarm', origin = "lower",animated=True, extent=[0,2*np.pi,0,2*np.pi],vmax=1,vmin=-1)
# # plt.title("u: Reference", fontsize = 18)
# # plt.xlabel("x", fontsize = 16)
# # plt.colorbar()
# # plt.ylabel("y", fontsize = 16)
# # plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'], fontsize = 16)
# # plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'], fontsize = 16)
# # plt.show()

# # plt.figure()
# # plt.imshow(u2D_flow_phi1*a_sol[0],interpolation='bicubic', origin = "lower",cmap='coolwarm',animated=True, extent=[0,2*np.pi,0,2*np.pi],vmax=1,vmin=-1)
# # plt.title("u: First modes", fontsize = 18)
# # plt.xlabel("x", fontsize = 16)
# # plt.colorbar()
# # plt.ylabel("y", fontsize = 16)
# # plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'], fontsize = 16)
# # plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'], fontsize = 16)
# # plt.show()

# # plt.figure()
# # plt.imshow(u2D_flow_phi2*a_sol[1],interpolation='bicubic', origin = "lower",cmap='coolwarm',animated=True, extent=[0,2*np.pi,0,2*np.pi],vmax=1,vmin=-1)
# # plt.title("u: Second modes", fontsize = 18)
# # plt.xlabel("x", fontsize = 16)
# # plt.colorbar()
# # plt.ylabel("y", fontsize = 16)
# # plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'], fontsize = 16)
# # plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'], fontsize = 16)
# # plt.show()

# # plt.figure()
# # plt.imshow(u2D_flow_phi3*a_sol[2],interpolation='bicubic', origin = "lower",cmap='coolwarm',animated=True, extent=[0,2*np.pi,0,2*np.pi],vmax=0.1,vmin=-0.1)
# # plt.title("u: Third modes", fontsize = 18)
# # plt.xlabel("x", fontsize = 16)
# # plt.colorbar()
# # plt.ylabel("y", fontsize = 16)
# # plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'], fontsize = 16)
# # plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', 'π/2', 'π', '3π/2', '2π'], fontsize = 16)
# # plt.show()

























