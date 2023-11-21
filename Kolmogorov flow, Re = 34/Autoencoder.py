"""
From

Racca et al., 2022
Modelling spatiotemporal turbulent dynamics with the convolutional autoencoder echo state network

"""

#%%

import os
os.environ["OMP_NUM_THREADS"] = "1" #set cores for numpy
import numpy as np
import tensorflow as tf
import json
tf.get_logger().setLevel('ERROR') #no info and warnings are printed 
tf.config.threading.set_inter_op_parallelism_threads(1) #set cores for TF
tf.config.threading.set_intra_op_parallelism_threads(1)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
   tf.config.experimental.set_memory_growth(gpu, True)
import matplotlib.pyplot as plt
import h5py
import time
from pathlib import Path
import matplotlib as mpl
import matplotlib.animation as animation


#%%

## DEFINE FUNCTIONS AND CLASSES USED ##

def split_data(U, b_size, n_batches):
    
    '''
    Splits the data in batches. Each batch is created by sampling the signal with interval
    equal to n_batches
    '''
    data   = np.zeros((n_batches, b_size, U.shape[1], U.shape[2], U.shape[3]))    
    for j in range(n_batches):
        data[j] = U[::skip][j::n_batches].copy()

    return data

@tf.function #this creates the tf graph
def model(inputs, enc_mods, dec_mods, is_train=False):
    
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

@tf.function #this creates the tf graph
def train_step(inputs, enc_mods, dec_mods, train=True):
    
    """
    Trains the model by minimizing the loss between input and output
    """
    
    # autoencoded field
    decoded  = model(inputs, enc_mods, dec_mods, is_train=train)[-1]

    # loss with respect to the data
    loss     = Loss_Mse(inputs, decoded)

    # compute and apply gradients inside tf.function environment for computational efficiency
    if train:
        # create a variable with all the weights to perform gradient descent on
        # appending lists is done by plus sign
        varss    = [] #+ Dense.trainable_weights
        for enc_mod in enc_mods:
            varss  += enc_mod.trainable_weights
        for dec_mod in dec_mods:
            varss +=  dec_mod.trainable_weights
        
        grads   = tf.gradients(loss, varss)
        optimizer.apply_gradients(zip(grads, varss))
    
    return loss

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

def U_to_X_enc(U_input, N_1):
    
    X_flattened = np.zeros((U_input.shape[0], N_1[3]*N_1[2]*N_1[1]))
    
    for k in range(N_1[3]):
        for t in range(U_input.shape[0]):
          for j in range(U_input.shape[1]):
              for i in range(U_input.shape[2]):
                  X_flattened[t, i+j*U_input.shape[2] + U_input.shape[1]*U_input.shape[2]*k]=U_input[t,j,i,k]
         
    return X_flattened

#%%

## INITIAL VALUES, PARAMETERS AND DATA MANIPULATIONS ##

# Read data from files
np_file = np.load('../Kolmogorov_Re34.npz')
X = np_file['X']  # Data
dt = np_file['dt']
coord = np_file['coord']
lyap = 0.067
Re = np_file['Re']

gridX = len(coord)
gridY = len(coord)
nk = int((gridX - 1)/2)
#

data_len = X.shape[0]

# Bring back data 2D matrix form

U_flow = np.zeros([data_len, gridX, gridY, 2])

# split uu_matrix into u and v matrix
Xu_flow = X[:, :gridX*gridY]
Xv_flow = X[:, gridX*gridY:]

for j in range(data_len):
    for i in range(gridX):
       U_flow[j, i, :, 0] = Xu_flow[j, gridX*i: gridX*i + gridX]
       U_flow[j, i, :, 1] = Xv_flow[j, gridX*i: gridX*i + gridX]

#%%

## Set up the autoencoder

b_size      = 50    #batch_size
n_batches   = 500    #number of batches
val_batches = 100   #int(n_batches*0.2) # validation set size is 0.2 the size of the training set
skip        = 10

print('Train Data%  :',100*b_size*n_batches*skip/U_flow.shape[0]) #how much of the data we are using for training
print('Val   Data%  :',100*b_size*val_batches*skip/U_flow.shape[0])
print("b_size = ", b_size)
print("n_batches = ", n_batches)
print("val_batches = ", val_batches)
print("skip = ", skip)

# training data
U_tt        = np.array(U_flow[:b_size*n_batches*skip].copy())            #to be used for random batches
U_train     = split_data(U_tt, b_size, n_batches).astype('float32') #to be used for randomly shuffled batches
# validation data
U_vv        = np.array(U_flow[b_size*n_batches*skip:
                         b_size*n_batches*skip+b_size*val_batches*skip].copy())
U_val       = split_data(U_vv, b_size, val_batches).astype('float32')             
del U_vv, U_tt

#%%

## DEFINE THE MODEL ##

# we do not have pooling and upsampling, instead we use stride=2

lat_dep       = 1                          #latent space depth
n_fil         = [12,24,lat_dep]        #number of filters ecnoder
n_dec         = [24,12,6]             #number of filters decoder
ker_size      = [(3,3), (5,5), (7,7)]               #kernel sizes
N_parallel    = int(len(ker_size)  )                          #number of parallel CNNs for multiscale
N_layers      = int(len(n_fil)  )                     #number of layers in every CNN
act           = 'tanh'                     #activation function

pad_enc       = 'valid'         #no padding in the conv layer
pad_dec       = 'valid'
p_size        = [0,1,2]         #stride = 2 periodic padding size          
p_fin         = [1,2,3]         #stride = 1 periodic padding size
p_dec         = 1               #padding in the first decoder layer
p_crop        = U_flow.shape[1]      #crop size of the output equal to input size

#initialize the encoders and decoders with different kernel sizes    
enc_mods      = [None]*(N_parallel)
dec_mods      = [None]*(N_parallel)    
for i in range(N_parallel):
    enc_mods[i] = tf.keras.Sequential(name='Enc_' + str(i))
    dec_mods[i] = tf.keras.Sequential(name='Dec_' + str(i))

#generate encoder layers    
for j in range(N_parallel):
    for i in range(N_layers):      

        #stride=2 padding and conv
        enc_mods[j].add(PerPad2D(padding=p_size[j], asym=True,
                                          name='Enc_' + str(j)+'_PerPad_'+str(i)))
        enc_mods[j].add(tf.keras.layers.Conv2D(filters = n_fil[i], kernel_size=ker_size[j],
                                      activation=act, padding=pad_enc, strides=2,
                        name='Enc_' + str(j)+'_ConvLayer_'+str(i)))

        #stride=1 padding and conv
        if i<N_layers-1:
            enc_mods[j].add(PerPad2D(padding=p_fin[j], asym=False,
                                                      name='Enc_'+str(j)+'_Add_PerPad1_'+str(i)))
            enc_mods[j].add(tf.keras.layers.Conv2D(filters=n_fil[i],
                                                    kernel_size=ker_size[j], 
                                                activation=act,padding=pad_dec,strides=1,
                                                    name='Enc_'+str(j)+'_Add_Layer1_'+str(i)))        

#%%

#explicitly obtain the size of the latent space
N_1      = enc_mods[-1](U_train[0]).shape
N_latent = N_1[-3]*N_1[-2]*N_1[-1]

#%%

#generate decoder layers            
for j in range(N_parallel):

    for i in range(N_layers):

        #initial padding of latent space
        if i==0: 
            dec_mods[j].add(PerPad2D(padding=p_dec, asym=False,
                                          name='Dec_' + str(j)+'_PerPad_'+str(i))) 
        
        #Transpose convolution with stride = 2 
        dec_mods[j].add(tf.keras.layers.Conv2DTranspose(filters = n_dec[i],
                                       output_padding=None,kernel_size=ker_size[j],
                                      activation=act, padding=pad_dec, strides=2,
                            name='Dec_' + str(j)+'_ConvLayer_'+str(i)))
        
        #Convolution with stride=1
        if  i<N_layers-1:       
            dec_mods[j].add(tf.keras.layers.Conv2D(filters=n_dec[i],
                                        kernel_size=ker_size[j], 
                                       activation=act,padding=pad_dec,strides=1,
                                      name='Dec_' + str(j)+'_ConvLayer1_'+str(i)))

    #crop and final linear convolution with stride=1
    dec_mods[j].add(tf.keras.layers.CenterCrop(p_crop + 2*p_fin[j],
                                                   p_crop+ 2*p_fin[j],
                            name='Dec_' + str(j)+'_Crop_'+str(i)))
    dec_mods[j].add(tf.keras.layers.Conv2D(filters=U_flow.shape[3],
                                            kernel_size=ker_size[j], 
                                            activation='linear',padding=pad_dec,strides=1,
                                              name='Dec_' + str(j)+'_Final_Layer'))

#%%

# run the model once to print summary
enc0, dec0 = model(U_train[0], enc_mods, dec_mods)
print('latent   space size:', N_latent)
print('physical space size:', U_flow[0].flatten().shape)
print('')

#%%

for j in range(N_parallel):
    enc_mods[j].summary()

for j in range(N_parallel):
    dec_mods[j].summary()

#%% 

# Now its time to train the model

plt.rcParams["figure.figsize"] = (15,4)
plt.rcParams["font.size"]  = 20

rng = np.random.default_rng() #random generator for later shuffling

Loss_Mse    = tf.keras.losses.MeanSquaredError()

n_epochs    = 2001 #number of epochs

#define optimizer and initial learning rate   
optimizer  = tf.keras.optimizers.legacy.Adam(amsgrad=True) #amsgrad True for better convergence
l_rate     = 0.002
optimizer.learning_rate = l_rate

lrate_update = True #flag for l_rate updating
lrate_mult   = 0.75 #decrease by thiscd factore the l_rate 
N_lr         = 20  #number of epochs before which the l_rate is not updated

# quantities to check and store the training and validation loss and the training goes on
old_loss      = np.zeros(n_epochs) #needed to evaluate training loss convergence to update l_rate
tloss_plot    = np.zeros(n_epochs) #training loss
vloss_plot    = np.zeros(n_epochs) #validation loss
old_loss[0]  = 1e6 #initial value has to be high
N_check      = 5   #each N_check epochs we check convergence and validation loss
patience     = 200 #if the val_loss has not gone down in the last patience epochs, early stop
last_save    = patience

t            = 1 # initial (not important value) to monitor the time of the training

foldername = "AE_%gC_%gL_e%g_%g_%g_d%g_%g_%g_N%g_%g_%g" %(N_parallel, N_layers, n_fil[0], n_fil[1], n_fil[2],n_dec[0], n_dec[1],n_dec[2], N_1[1], N_1[2], N_1[3])
path = './'+foldername+'/Kflow_RE34_AE' 

#%% 

for epoch in range(n_epochs):
    print(epoch)
    if epoch - last_save > patience: break #early stop
                
    #Perform gradient descent for all the batches every epoch
    loss_0 = 0
    rng.shuffle(U_train, axis=0) #shuffle batches
    for j in range(n_batches):
            loss    = train_step(U_train[j], enc_mods, dec_mods)
            loss_0 += loss
    
    #save train loss
    tloss_plot[epoch]  = loss_0.numpy()/n_batches     
    
    # every N epochs checks the convergence of the training loss and val loss
    if (epoch%N_check==0):
        
        #Compute Validation Loss
        loss_val = 0
        for j in range(val_batches):
            loss        = train_step(U_val[j], enc_mods, dec_mods,train=False)
            loss_val   += loss
        
        #save validation loss
        vloss_plot[epoch]  = loss_val.numpy()/val_batches 
        
        # Decreases the learning rate if the training loss is not going down with respect to 
        # N_lr epochs before
        if epoch > N_lr and lrate_update:
            #check if the training loss is smaller than the average training loss N_lr epochs ago
            tt_loss   = np.mean(tloss_plot[epoch-N_lr:epoch])
            if tt_loss > old_loss[epoch-N_lr]:
                #if it is larger, load optimal val loss weights and decrease learning rate
                print('LOADING MINIMUM')
                for i in range(N_parallel):
                    enc_mods[i].load_weights(path + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'_weights.h5')
                    dec_mods[i].load_weights(path + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'_weights.h5')

                optimizer.learning_rate = optimizer.learning_rate*lrate_mult
                optimizer.set_weights(min_weights)
                print('LEARNING RATE CHANGE', optimizer.learning_rate.numpy())
                old_loss[epoch-N_lr:epoch] = 1e6 #so that l_rate is not changed for N_lr steps
        
        #store current loss
        old_loss[epoch] = tloss_plot[epoch].copy()
        
        #save best model (the one with minimum validation loss)
        if epoch > 1 and vloss_plot[epoch] < \
                         (vloss_plot[:epoch-1][np.nonzero(vloss_plot[:epoch-1])]).min():
        
            #saving the model weights
            print('Saving Model..')
            Path(path).mkdir(parents=True, exist_ok=True) #creates directory even when it exists
            for i in range(N_parallel):
                enc_mods[i].save(path + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5')
                dec_mods[i].save(path + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5')
                enc_mods[i].save_weights(path + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'_weights.h5')
                dec_mods[i].save_weights(path + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'_weights.h5')
            
            #saving optimizer parameters
            min_weights = optimizer.get_weights()
            hf = h5py.File(path + '/opt_weights.h5','w')
            for i in range(len(min_weights)):
                hf.create_dataset('weights_'+str(i),data=min_weights[i])
            hf.create_dataset('length', data=i)
            hf.create_dataset('l_rate', data=optimizer.learning_rate)  
            hf.close()
            
            last_save = epoch #store the last time the val loss has decreased for early stop

        # Print loss values and training time (per epoch)
        print('Epoch', epoch, '; Train_Loss', tloss_plot[epoch], 
              '; Val_Loss', vloss_plot[epoch],  '; Ratio', (vloss_plot[epoch])/(tloss_plot[epoch]))
        print('Time per epoch', (time.time()-t)/N_check)
        print('')
        
        t = time.time()
        
    if (epoch%20==0) and epoch != 0:    
        #plot convergence of training and validation loss (to visualise convergence during training)
        plt.figure(epoch)
        plt.title('MSE convergence')
        plt.yscale('log')
        plt.grid(True, axis="both", which='both', ls="-", alpha=0.3)
        plt.plot(tloss_plot[np.nonzero(tloss_plot)], 'y', label='Train loss')
        plt.plot(np.arange(np.nonzero(vloss_plot)[0].shape[0])*N_check,
                 vloss_plot[np.nonzero(vloss_plot)], label='Val loss')
        plt.xlabel('epochs')
        plt.legend()    
        plt.tight_layout()
        plt.savefig('./'+foldername+'/MSE_convergence.pdf', dpi = 300)
        plt.show()

#%%

#Now we need encoded data to train the ESN, which will now be made

N_pos     = 2417 #split in k interval of N_pos length needed to process long timeseries
k         = int(data_len/N_pos)

N_x      = U_flow.shape[1]
N_y      = U_flow.shape[2]
Latents  = [N_latent]

#%%

for N_latent in Latents:
   
   enc = [None]*N_parallel
   dec = [None]*N_parallel
   for i in range(N_parallel):
       enc[i] = tf.keras.models.load_model(path + '/enc_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5', 
                                             custom_objects={"PerPad2D": PerPad2D})
   for i in range(N_parallel):
       dec[i] = tf.keras.models.load_model(path + '/dec_mod'+str(ker_size[i])+'_'+str(N_latent)+'.h5',
                                             custom_objects={"PerPad2D": PerPad2D})
   #%%
   U_enc = np.zeros((data_len, N_1[1], N_1[2], N_1[3]))
   #encode all the data to provide time series in latent space for the ESN
   for i in range(k):
       U_enc[i*N_pos:(i+1)*N_pos]= model(U_flow[i*N_pos:(i+1)*N_pos], enc, dec)[0]

    # Encode last data points outside a even multiple of N_pos

   U_enc[k*N_pos:] = model(U_flow[k*N_pos:], enc, dec)[0]

   # Now flatten the encoded data similar to what was done with the reference data 

   X_enc = np.zeros((data_len, N_1[1]*N_1[2]*N_1[3]))
   X_enc = U_to_X_enc(U_enc, N_1)

   foldername = "AE_%gC_%gL_e%g_%g_%g_d%g_%g_%g_N%g_%g_%g" %(N_parallel, N_layers, n_fil[0], n_fil[1], n_fil[2],n_dec[0], n_dec[1],n_dec[2], N_1[1], N_1[2], N_1[3])
   fln = './'+foldername+'/Encoded_data_Re' + str(Re)
   np.savez(fln, U_enc = X_enc, Re = Re, dt = dt, Lyap = lyap, coord = coord, n_epochs = n_epochs, N_parallel = N_parallel, ker_size = ker_size, N_latent=N_latent, b_size = b_size, n_batches = n_batches, skip = skip, N_1 = N_1)

