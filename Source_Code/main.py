import theano, sys
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, RecurrentLayer, LSTMLayer, DenseLayer, SliceLayer, get_all_params
from lasagne.updates import adagrad, adam
from lasagne.nonlinearities import sigmoid, tanh, softmax
from lasagne.objectives import binary_crossentropy, categorical_crossentropy
import numpy as np
import pickle
import time
from helpers_to_main import *

t0 = time.time()
# Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(1))

# Declare Global Variables
NUM_FEATURES = None
X = None
Y = None
SEQ_LENGTH = 32
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2000
PRINT_FUGUES = 25
GRAD_CLIPPING = 100
NUM_NEURONS = 64
NUM_LAYERS = 10
data_size = BATCH_SIZE*228

np.random.seed(0)
print "\n\n-------------------------------"
print "Loading pitch_matrix ..."
sh = 0
pitch_matrix = pickle.load(open('training_data/pitch_matrix_'+str(96/4)+'ticks_sh'+"%02d" % (sh,)+'.p', 'rb'))
# at this point, pitch_matrix is a list of 2D lists. 
# pitch_matrix[i] is a fugue in the training set, for each i.

# flatten pitch_matrix into a single 2D ndarray
pitch_matrix = flatten_pitch_matrix(pitch_matrix)
NUM_FEATURES = pitch_matrix.shape[1]  # 61
 
# extract only lowest notes
pitch_matrix = make_monophonic(pitch_matrix) 
# clip pitch_matrix for testing purposes.
if data_size < pitch_matrix.shape[0] - SEQ_LENGTH:
	clipped_pitch_matrix_length = data_size + SEQ_LENGTH 
	print 'pitch_matrix original, pitch_matrix.shape = {}'.format(pitch_matrix.shape)
	pitch_matrix = pitch_matrix[:clipped_pitch_matrix_length]
	print 'pitch_matrix CLIPPED,  pitch_matrix.shape = {}'.format(pitch_matrix.shape)
else:
	print '!!!!!!pitch_matrix NOT clipped, pitch_matrix.shape = {}'.format(pitch_matrix.shape)


# print the network architecture
print '\n\t-----------------------------\n'
for _ in range(3):
	print ' '*12 + '       ' + 'O      '*NUM_LAYERS + ' '
for _ in range(4):
	print ' '*12 + 'O      ' + 'O      '*NUM_LAYERS + 'O'
for _ in range(3):
	print ' '*12 + '       ' + 'O      '*NUM_LAYERS + ' '
print ' '*12 + " "
unit_no = "%3d" % (NUM_NEURONS,) + " "*4 
print ' '*12+"%2d" % (NUM_FEATURES,) + " "*4 + unit_no*NUM_LAYERS + " %2d" % (NUM_FEATURES,)  

print '\n\t-----------------------------'
print '\tSEQ_LENGTH = {}'.format(SEQ_LENGTH)
print '\tBATCH_SIZE = {}'.format(BATCH_SIZE) 
print '\tLEARNING_RATE = {}'.format(LEARNING_RATE)
print '\tdata_size = {}'.format(data_size)
print '\tPRINT_FUGUES = {}'.format(PRINT_FUGUES)
print '\t-----------------------------\n'

'''||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def next_note(x):
	predict = probabilities(x)
	ix = np.argmax(predict)
	note = np.zeros((1,NUM_FEATURES))
	note[0,ix] = 1
	return note

def generate_a_fugue(epoch, loss, N=16*32):
	'''
	INPUT:  epoch and loss: Passed just to name the output files.	
	N: How many steps the fugue is to be generated for. 
	'''
	
	# Advance the RNN model for N steps at its current state.
	fugue = np.zeros((N, NUM_FEATURES))
	
	x = np.zeros_like(x_seed)
	x[:] = x_seed[:]
	
	for i in xrange(N):		
		fugue[i] = next_note(x)
		x[0, :-1, :] = x[:, 1:, :] 
		x[0, -1, :] = fugue[i, :] 

	fugue = np.vstack((x_seed[0],fugue))
	# # # # # # # # # # # # # # # # # # 
	
	print_epoch = "%04d" % (epoch,)
	print_SEQ = "%02d" % (SEQ_LENGTH,)
	print_LSTM = "%01d" % (NUM_LAYERS,)+"x"+"%03d" % (NUM_NEURONS,)
	
	filename = 'fugue' + \
			'_SEQ' + print_SEQ + \
			'_LSTM' + print_LSTM + \
			'_epoch' + print_epoch

	dump_dict = {'NUM_FEATURES' : NUM_FEATURES,
				'SEQ_LENGTH' : SEQ_LENGTH,
				'BATCH_SIZE' : BATCH_SIZE,
				'LEARNING_RATE' : LEARNING_RATE,
				'NUM_EPOCHS' : NUM_EPOCHS,
				'GRAD_CLIPPING' : GRAD_CLIPPING,
				'NUM_NEURONS' : NUM_NEURONS,
				'NUM_LAYERS' : NUM_LAYERS,
				'data_size' : data_size,
				'loss' : loss,
				'fugue' : fugue}
	
	pickle.dump(dump_dict, open('Synthesized_Fugues/'+filename+'.p', 'wb'))
	time_series = pitch_matrix_TO_time_series_legato(fugue, sh=24)
	
	if len(time_series) > 0:
		time_series_TO_midi_file(time_series, 'Synthesized_Fugues/'+filename+'.mid')
	elif len(time_series) == 0:
		print 'ZERO NOTES PREDICTED'
	
	return
	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def build_rnn(sequence_length=SEQ_LENGTH, num_units=NUM_NEURONS):
	# input layer
	# generate the data to pass into this
	l_in = InputLayer(shape=(None, sequence_length, NUM_FEATURES))
	l_LSTMs = [LSTMLayer(l_in, num_units=num_units, 
						grad_clipping=GRAD_CLIPPING, nonlinearity=tanh)]

	for i in range(1,NUM_LAYERS):
		l_LSTMs.append(LSTMLayer(l_LSTMs[i-1], num_units=num_units, 
		grad_clipping=GRAD_CLIPPING, nonlinearity=tanh))

	# The l_forward layer creates an output of dimension (batch_size, SEQ_LENGTH, N_HIDDEN)
	# Since we are only interested in the final prediction, we isolate that quantity and feed it to the next layer. 
	# The output of the sliced layer will then be of size (batch_size, N_HIDDEN)
	'''l_slice = SliceLayer(l_LSTM2, -1, 1)''' # don't use this SliceLayer
	
	# output layer
	l_out = DenseLayer(l_LSTMs[-1], num_units=NUM_FEATURES, W=lasagne.init.Normal(), nonlinearity=softmax)
	
	return l_in, l_out
	
'''||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''
#x_seed_from_training = np.zeros((1, SEQ_LENGTH, NUM_FEATURES))
#x_seed_from_training[0] = pitch_matrix[:SEQ_LENGTH]
#x_seed = np.asarray([val for i,val in enumerate(x_seed_from_training[0,:,:]) if i%2 == 0])

x_seed = np.zeros((1, SEQ_LENGTH, NUM_FEATURES))
for i, val in enumerate(pitch_matrix[:SEQ_LENGTH/2,:]):
	x_seed[0,i,:] = val
	x_seed[0,i*2,:] = val


print "Building network ..."
num_of_params = NUM_FEATURES * NUM_NEURONS*2 + (NUM_NEURONS + NUM_FEATURES) \
					+ (NUM_LAYERS-1) * (NUM_NEURONS**2) + (NUM_NEURONS*(NUM_LAYERS-1))
print 'total no of params to be learned:{}'.format(num_of_params)

l_in, l_out = build_rnn()
target_values = T.imatrix('target_output')
network_output = lasagne.layers.get_output(l_out)

# The loss function of predictions and target.
loss = T.mean(categorical_crossentropy(network_output, target_values))

# Retrieve all parameters from the network
all_params = get_all_params(l_out, trainable=True)

# Compute AdaGrad updates for training
print "Computing updates for training w/ adam..."
updates = adam(loss_or_grads=loss, params=all_params, learning_rate=LEARNING_RATE)

# Theano functions for training and computing loss
print "Compiling functions ..."
train_function = theano.function([l_in.input_var, target_values], loss, updates=updates, allow_input_downcast=True)

# Need probability distribution of the next note given 
# the state of the network and the input_seed.
# Compile a theano function called probabilities to produce 
# the probability distribution of the prediction.

# `probabilities` is analogous to calling .predict()
probabilities = theano.function([l_in.input_var], network_output, allow_input_downcast=True)

'''  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || 
     \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/   
	   T  R  A  I  N  I  N  G  --  T  R  A  I  N  I  N  G  --  T  R  A  I  N  I  N  G
'''
# make the training data tensors X and Y, and train the model on them
print "Training ..."
X, Y = make_X_Y(pitch_matrix, data_size, SEQ_LENGTH, NUM_FEATURES)

p, avg_loss, dummy_previous, loss_p = 0, 0, -1, []
for it in xrange(1, data_size * NUM_EPOCHS / BATCH_SIZE + 1):
	epoch = float(it) * float(BATCH_SIZE) / data_size
	#sys.stdout.flush();	print "it:", it, ", epoch:", round(epoch,3),"        \r",
	x, y = make_batch(p, X, Y, data_size, BATCH_SIZE)
	avg_loss += train_function(x, y)
	
	
	p = p + BATCH_SIZE
	dummy_previous = epoch % PRINT_FUGUES
	
	if epoch == 1.0:
		print "At epoch:", round(epoch,3),", avg_loss=", avg_loss

	if p == data_size :
		# Another epoch completed.
		# print "epoch {} completed. Reset: avg_loss=0, p=0".format(epoch)
		sys.stdout.flush()
		print "At epoch:", round(epoch,3),", avg_loss=", avg_loss," "*99,"\r",
		time.sleep(1)
		if epoch % PRINT_FUGUES == 0.0:
			# generate a fugue every PRINT_FUGUES epochs
			generate_a_fugue(epoch=epoch, loss=loss_p) 
			print "At epoch:", round(epoch,3),", avg_loss=", avg_loss,", .mid and .p files saved. t =", round(time.time()-t0),"sec."
	
		loss_p.append(avg_loss)
		p = 0
		avg_loss = 0

	

