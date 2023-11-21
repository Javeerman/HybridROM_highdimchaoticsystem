"""
Implementation of the Echo State Network in Tensorflow as obtained from

Lesjak and Doan, 2021: 
Chaotic systems learning with hybrid echo state network/proper orthogonal decomposition based model

The general structure and code for the echo state network
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers.recurrent import _generate_zero_filled_state_for_cell
tf.keras.backend.set_floatx('float64')
#tf.config.experimental_run_functions_eagerly(True)

#%%

class ESNCell(Layer):
    """
    Base cell for ESN. Implementation according to: A Practical Guide to Applying Echo State Networks, Mantas Lukosevicius
    Base cell is meant to get wrapped with tf.keras.layers.RNN(cell) in order to transfer internal states for recurrency.

    :param num_units: Number of internal units = Reservoir size
    :param num_inputs: Number of input and output features
    :param alpha: Leaking rate (0,1]. alpha=1 special case without leaky integration.
    :param rho: Spectral radius of Echo State Matrix. rho<1 ensures echo state property, however bigger values may increase accuracy
    :param sparseness: Sparsity of the Reservoir (= makes most elements of input matrix Win equal to zero)
    :param sigma_in: Lower and upper boundary for uniform distribution in input matrix Win
    :param rng: None or np.random.RandomState(seed) with seed for reproducability
    :param activation: Activation function
    """

    def __init__(self, num_units, num_inputs=3, alpha=0.1, rho=0.6,
                 sparseness=0.0,
                 sigma_in=1.0,
                 rng=None,
                 activation='tanh',
                 **kwargs):
        super(ESNCell, self).__init__(**kwargs)

        # fixed variables
        self._num_units = num_units
        self._activation = activations.get(activation)
        self._num_inputs = num_inputs


        # variables that potentially can be changed/optimized (for future version)
        self.alpha = alpha
        self.rho = rho  # rho has to be <1. to ensure the echo state property (see [2])
        self.sparseness = sparseness
        self.sigma_in = sigma_in

        # Random number generator initialization
        self.rng = rng
        if rng is None:
            self.rng = np.random.RandomState()
            
        # build initializers for tensorflow variables
        self.win = self.buildInputMatrix()
        self.wecho = self.buildEchoMatrix()

        # convert the weight to tf variable
        #self.Win = tf.get_variable('Win', initializer=self.win, trainable=False)
        #self.Wecho = tf.get_variable('Wecho', initializer=self.wecho, trainable=False)

        self.setEchoStateProperty()

        self.state_size = self._num_units
        self.output_size = self._num_units
        print('ESN successfully initializied with num_units=%d, num_inputs=%d, alpha=%.2f, rho=%.2f, sigma_in=%.2f' %(num_units,num_inputs,alpha,rho,sigma_in))

    def build(self, input_shape):
        """
        Creates variables at first call. (for instance when calling the cell(Input) with Input layer that specifies the input_shape)

        :param input_shape:
        """

        # Create a trainable weight variable for this layer.
        self.Wout = self.add_weight(name='Wout',
                                      shape=(self._num_inputs, self.state_size), #TODO make sure transposed is ok
                                      initializer=tf.keras.initializers.RandomUniform(minval=-1, maxval=1, seed=1),
                                      trainable=True)

        self.Wecho = K.constant(self.wecho, dtype='float64',name='Wecho')
        self._non_trainable_weights.append(self.Wecho)

        self.Win = K.constant(self.win, dtype='float64', name='Win')
        self._non_trainable_weights.append(self.Win)

        self.unity = K.eye(self.state_size)


        print('build weights')
        super(ESNCell, self).build(input_shape)  # Be sure to call this at the end


    def call(self, inputs, states):

        """
        Gets called every timestep of K.rnn loop in recurrent layer, that iterates over all timesteps.
        Echo-state RNN:
                    x = x + h*(f(W*inp + U*g(x)) - x).

        :param inputs: Input at current timestep
        :param states: Internal state of previous timestep
        """
        state = states[0]

        new_state = state + self.alpha * (
                self._activation(
                    K.dot(inputs, self.Win) +
                    K.dot(state, self.Wecho)
                )
                - state)

        output = new_state

        return output, [new_state] #for multiple states see constructor state_size and LSTM


    #def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    #    print('zero state generated')
    #    return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)


    def setEchoStateProperty(self):
        """ optimize U to obtain alpha-improved echo-state property """
        # I know it's stupid for the time being but it is a placeholder for future treatment of the matrix
        # (potential meta-optimization and other)
        self.wecho = self.normalizeEchoStateWeights(self.wecho)

    # construct the Win matrix (dimension num_inputs x num_units)
    def buildInputMatrix(self):
        """
            Returns:

            Matrix representing the
            input weights to an ESN
        """

        # Input weight matrix initializer according to [3,4]
        # Each unit is connected randomly to a given input with a weight from a uniform distribution

        # without bias at the input
        # W = np.zeros((self._num_inputs,self._num_units))
        # for i in range(self._num_units):
        # W[self.rng.randint(0,self._num_inputs),i] = self.rng.uniform(-self.sigma_in,self.sigma_in)

        # Added bias in the input matrix
        W = np.zeros((self._num_inputs, self._num_units))
        for i in range(self._num_units):
            W[self.rng.randint(0, self._num_inputs), i] = self.rng.uniform(-self.sigma_in, self.sigma_in)

        # Dense input weigth [input] as in [1,2]
        # Input weigth matrix [input]
        # W = self.rng.uniform(-self.sigma_in, self.sigma_in, [self.num_inputs, self._num_units]).astype("float64")

        # Dense input weigth [bias, input] (as in [1,2])
        # W = self.rng.uniform(-self.sigma_in, self.sigma_in, [self.num_inputs+1, self._num_units]).astype("float64")

        return W.astype('float64')

    def getInputMatrix(self):
        return self.win

    def buildEchoMatrix(self):
        """
            Returns:

            A 1-D tensor representing the
            inner weights to an ESN (to be optimized)
        """

        # Inner weight tensor initializer
        # 1) Build random matrix from normal distribution between [0.,1.]
        # W = self.rng.randn(self._num_units, self._num_units).astype("float64") * \
        # (self.rng.rand(self._num_units, self._num_units) < (1. - self.sparseness) )

        # 2) Build random matrix from uniform distribution
        W = self.rng.uniform(-1.0, 1.0, [self._num_units, self._num_units]).astype("float64") * (
                    self.rng.rand(self._num_units, self._num_units) < (
                        1. - self.sparseness))  # trick to add zeros to have the sparseness required
        return W

    def normalizeEchoStateWeights(self, W):

        # compute the spectral radius of these weights:
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        # rescale them to reach the requested spectral radius:
        W = W * (self.rho / radius)

        return W.astype('float64')

    def getEchoMatrix(self):
        return self.wecho


    #def compute_output_shape(self, input_shape):
    #    return (input_shape[1], self.output_dim)


#%%

class ESN():

  """
  ESN Model class. Implementation according to: A Practical Guide to Applying Echo State Networks, Mantas Lukosevicius

  :param num_units: Number of internal units = Reservoir size
  :param num_inputs: Number of input and output features
  :param alpha: Leaking rate (0,1]. alpha=1 special case without leaky integration.
  :param rho: Spectral radius of Echo State Matrix. rho<1 ensures echo state property, however bigger values may increase accuracy
  :param sparseness: Sparsity of the Reservoir (= makes most elements of input matrix Win equal to zero)
  :param sigma_in: Lower and upper boundary for uniform distribution in input matrix Win
  :param rng: None or np.random.RandomState(seed) with seed for reproducability
  :param activation: Activation function
  :param beta: regularization parameter (see ridge regression in training)

  """

  def __init__(self, num_units, num_inputs=3, alpha=0.1, rho=0.6,
                 sparseness=0.0,
                 sigma_in=1.0,
                 rng=None,
                 activation='tanh',
                 beta = 0.001,
                 **kwargs):
    input = tf.keras.Input(shape=(None,num_inputs),batch_size=1)
    cell = tf.keras.layers.RNN(ESNCell(num_units, num_inputs, alpha, rho, sparseness,
                                            sigma_in,
                                            rng,
                                            activation),
                               return_sequences=True,name='ESNCell',stateful=True,**kwargs)
    output = cell(input) #builds weight matrices (first call, calls build function)
    self.network = tf.keras.Model(input,output)
    #self.network.compile(loss='mse', optimizer='adam')

    #regularization parameter
    self.beta = beta

  def summary(self):
      return self.network.summary()

  def predict(self,inputs):
      return self.predict_keras(inputs).numpy()


  @tf.function
  def predict_keras(self, inputs):
      """
      Predicts values for the given inputs using the internal state from the last prediction as initial state.
      Use reset_states() or set_states(states) to modify the initial state.

      :param inputs: Inputs in shape (1, timesteps, features)
      :return: numpy array with predicted values. shape (timesteps,features)
      """

      states = self.network(inputs) #TODO shape correction

      #shape correction (to be neglected in updated version by smartly changing dimensions)
      states = K.transpose(states[0,:,:]) #(nx,t)

      #apply quadratic trafo (T1 algorithm)
      states2 = K.square(states)
      new_state = K.reshape(states[0,:],shape=(1,states.shape[1]))
      for i in range(1,states.shape[0]):
          if (np.mod(i, 2) != 0):
              new_state = K.concatenate((new_state,K.reshape(states2[i,:],shape=(1,states.shape[1]))),axis=0)
          else:
              new_state = K.concatenate((new_state,K.reshape(states[i,:],shape=(1,states.shape[1]))),axis=0)

      Wout = self.network.get_layer('ESNCell').cell.Wout #.numpy() #(nu,nx)

      output = K.transpose(K.dot(Wout,new_state))


      return output

  @tf.function
  def fit(self,inputs, targets, washout): #targets(t,nu)
      """
      Trains the ESN Network and resets the state to zero afterwards.

      :param inputs: Inputs in shape (1, timesteps, features)
      :param targets: Target Values with shape (timesteps, features)
      :param washout: Number of steps to be removed at the beginning (Initial transients of reservoir)
      """

      states_all = self.network(inputs)
      states_all = K.transpose(states_all[0,:,:]) #(nx,t)
      #inputs = np.transpose(inputs[0,:,:]) #(nu,t)
      targets = K.transpose(targets) #(nu,t)

      #apply quadratic trafo (T1 algorithm)

      states2 = K.square(states_all)
      #K.print_tensor('squared states')
      #K.print_tensor(states2)

      new_state = K.reshape(states_all[0,:],shape=(1,states_all.shape[1]))
      for i in range(1,states_all.shape[0]):
          if (np.mod(i, 2) != 0):
              new_state = K.concatenate((new_state,K.reshape(states2[i,:],shape=(1,states_all.shape[1]))),axis=0)
          else:
              new_state = K.concatenate((new_state,K.reshape(states_all[i,:],shape=(1,states_all.shape[1]))),axis=0)

      X = new_state

      #Washout
      X = X[:,washout:]
      targets = targets[:,washout:]


      Wout = K.dot(K.dot(targets,K.transpose(X)),tf.linalg.inv(K.dot(X,K.transpose(X)) + self.beta * self.network.get_layer('ESNCell').cell.unity))
      #TODO find best beta by selecting beta in a way that beta yields optimal output

      print('Wout computed')
      #print(Wout)

      K.update(self.network.get_layer('ESNCell').cell.Wout,Wout)
      #K.set_value(self.network.get_layer('ESNCell').cell.Wout, Wout.numpy())
      print('Wout stored')

      print('states got reset after training')
      self.reset_states()

      #if washout != 0:
      #    print('set state to washout state')
      #    self.set_states(states_all[:,washout-1].reshape((1,states_all.shape[0])))
      #else:
      #    print('states got reset after training')
      #    self.reset_states()
      #TODO instead of reset_state --> set state to washout value.

  def reset_states(self):
      """
      Resets the internal ESN state to zero

      """
      print('states reset')
      return self.network.reset_states()

  def set_states(self, state):
      """
      Resets the internal ESN state to zero if no param is given, otherwise sets the state to param

      :param state: numpy array with shape (1,num_units)
      """
      return self.network.get_layer('ESNCell').reset_states(state)

  def save_weights(self,filepath):
      """
      Saves the internal weight matrices

      :param filepath: Filepath. example: /home/Workdir/weights.h5
      """
      return self.network.save_weights(filepath)

  def load_weights(self,filepath):
      """
      Saves the internal weight matrices

      :param filepath: Filepath. example: /home/Workdir/weights.h5
      """
      return self.network.load_weights(filepath)

  def save(self,filepath):
      """
      Saves the whole model.

      :param filepath: Filepath. example: /home/Workdir/weights.h5
      :return:
      """
      return self.network.save(filepath)



# def main():

#     import os
#     import matplotlib.pyplot as plt
#     import numpy.linalg as LA

#     n_neurons = 500
#     rng = np.random.RandomState(1)
#     degree = 4.11  # average number of connections of a unit to other units in reservoir (3.0)
#     sparseness = 1. - degree / (n_neurons - 1.)  # sparseness of W

#     model = ESN(n_neurons, 3, 1.0, 0.405, sparseness, 0.9875, rng, 'tanh',0.000067) #0.000067


#     #inp = np.ones((1, 4, 3))
#     #inp[:, 1, :] = inp[:, 1, :] * 2
#     #inp[:, 2, :] = inp[:, 2, :] * 3
#     #inp[:, 3, :] = inp[:, 3, :] * 4
#     #out = np.ones((4,3))
#     #out[0,:] = out[1,:]*2
#     #out[1,:] = out[1,:]*3
#     #out[2,:] = out[2,:]*4
#     #out[3,:] = out[3,:]*5


#     #model.fit(inp,out)
#     #print('first prediction')
#     #print(model.predict(inp))
#     #model.reset_states()
#     #model.set_states(np.zeros((1,200)))
#     #print('second prediction')
#     #print(model.predict(inp[:,0,:].reshape((1,1,3))))
#     #print(model.predict(inp[:, 1, :].reshape((1, 1, 3))))
#     #print(model.predict(inp[:, 2, :].reshape((1, 1, 3))))
#     #print(model.predict(inp[:, 3, :].reshape((1, 1, 3))))

# #############################################################################################


#     epsilon = 0.2
#     normalization = 0
#     # Read data from files
#     os.chdir("/Users/ml/Documents/Studium/Masterarbeit/Code/pemlm/LorenzSys/")
#     np_file = np.load('./Lorenz_data.npz')
#     X = np_file['X']  # Data
#     dt = np_file['dt']
#     t_split = np_file['t_stop_train']
#     t_skip = np_file['t_skip']
#     val_ratio = np_file['val_ratio']
#     Lyap = np_file['Lyap']

#     # prepare data for training

#     # skip beginning
#     i_skip = int(t_skip / dt)
#     X = X[i_skip:, :]

#     # normalize data
#     stddev = np.std(X, axis=0)
#     avg = np.average(X, axis=0)

#     if normalization == 0:
#         X = (X - avg) / stddev

#     if normalization == 1:
#         X = X - avg
#         max = np.array([np.amax(np.absolute(X[:, 0])), np.amax(np.absolute(X[:, 1])), np.amax(np.absolute(X[:, 2]))])
#         X = X / max

#     input_all = X[:-1, :]
#     output_all = X[1:, :]

#     # split data into training, validation
#     idx_split = int(t_split / dt) - i_skip
#     assert idx_split > 0, 'skip time is bigger than split time'
#     # index that seperates training and validation data
#     idx_val = int(idx_split * (1 - val_ratio))

#     input_train = input_all[:idx_split, :]
#     output_train = output_all[:idx_split, :]

#     #training_time = 10
#     #training_idx = int(training_time/dt)
#     resync_time = 1
#     resync_idx = int(resync_time/dt)
#     washout = 100

#     #shorten training data
#     #input_train = input_train[:training_idx,:]
#     #output_train = output_train[:training_idx,:]

#     #reshape input
#     input_train = input_train.reshape((1,input_train.shape[0],input_train.shape[1]))
#     print(input_train.shape)

#     model.fit(input_train,output_train,washout)
#     model.reset_states()

#     #Teacher forced prediction
#     #Y = model.predict(input_train[:,washout:,:])


#     #plt.plot(input_train[0, washout:, 0])
#     #plt.plot(Y[:, 0], '--')
#     #plt.show()

#     #plt.plot(input_train[0, washout:, 1])
#     #plt.plot(Y[:, 1], '--')
#     #plt.show()

#     #plt.plot(input_train[0, washout:, 2])
#     #plt.plot(Y[:, 2], '--')
#     #plt.show()

#     #model.reset_states()



#     t_pred = 10 / Lyap
#     idx_end = int(t_pred / dt)
#     x_init = X[idx_split-resync_idx:idx_split-1,:]
#     x_ref = X[idx_split:, :]
#     assert idx_end < x_ref.shape[0], 't_pred too long'
#     x_ref = x_ref[:idx_end, :]

#     #initialize reservoir
#     x_init = x_init.reshape((1,x_init.shape[0],x_init.shape[1]))
#     model.predict(x_init)

#     Y = []

#     y_last = x_ref[0,:].reshape((1, 1, 3))
#     Y.append(x_ref[0,:].reshape((1,3)))

#     print('start natural response')
#     for i in range(x_ref.shape[0] - 1):
#         Y.append(model.predict(y_last))
#         y_last = Y[-1].reshape((1, 1, 3))
#         print('[%d/%d] predicitions done' % ((i),x_ref.shape[0])) if i%100 == 0 else None

#     Y = np.array(Y)
#     Y = Y.reshape(Y.shape[0], Y.shape[2])


#     # denormalize
#     if normalization == 0:
#         Y = Y * stddev + avg
#         x_ref = x_ref * stddev + avg

#     if normalization == 1:
#         Y = Y * max + avg
#         x_ref = x_ref * max + avg

#     # calculate Error
#     err = LA.norm(Y[:, :] - x_ref[:, :], axis=1) / np.sqrt(
#         np.average(np.square(LA.norm(x_ref[:, :], axis=1))))
#     plt.plot(err)
#     plt.show()

#     #plt.plot(x_ref[:, 0])
#     #plt.plot(Y[:, 0], '--')
#     #plt.show()

#     #plt.plot(x_ref[:, 1])
#     #plt.plot(Y[:, 1], '--')
#     #plt.show()

#     #plt.plot(x_ref[:, 2])
#     #plt.plot(Y[:, 2], '--')
#     #plt.show()


# if __name__ == '__main__':
#     main()