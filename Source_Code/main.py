import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, RecurrentLayer, LSTMLayer, DenseLayer, SliceLayer, get_all_params
from lasagne.updates import adagrad
from lasagne.nonlinearities import sigmoid, tanh, softmax
from lasagne.objectives import binary_crossentropy, categorical_crossentropy
import numpy as np
import pickle
import time
from helpers_to_main import *

t0 = time.time()
#Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(1))

# Declare Global Variables
NUM_FEATURES = None
X = None
Y = None
SEQ_LENGTH = 32
BATCH_SIZE = 256
LEARNING_RATE = 5e-3
NUM_EPOCHS = 2000
PRINT_FUGUES = 20
GRAD_CLIPPING = 100
NUM_UNITS = 256
NUM_LAYERS = 3
data_size = BATCH_SIZE*128

np.random.seed(0)
print "\n\n\n\n\n\n==================================================="
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
print ' '*12
for _ in range(3):
	print ' '*12 + '       ' + 'O      '*NUM_LAYERS + ' '
for _ in range(4):
	print ' '*12 + 'O      ' + 'O      '*NUM_LAYERS + 'O'
for _ in range(3):
	print ' '*12 + '       ' + 'O      '*NUM_LAYERS + ' '
print ' '*12 + " "
unit_no = "%03d" % (NUM_UNITS,) + " "*4 
print ' '*12+"%02d" % (NUM_FEATURES,) + " "*4 + unit_no*NUM_LAYERS + "%02d" % (NUM_FEATURES,)  
print ' '*12

print '\n\t-----------------------------'
print '\tSEQ_LENGTH = {}'.format(SEQ_LENGTH)
print '\tBATCH_SIZE = {}'.format(BATCH_SIZE) 
print '\tNUM_UNITS = {}'.format(NUM_UNITS)
print '\tNUM_LAYERS = {}'.format(NUM_LAYERS)
print '\tLEARNING_RATE = {}'.format(LEARNING_RATE)
print '\tdata_size = {}'.format(data_size)
print '\tPRINT_FUGUES = {}'.format(PRINT_FUGUES)
print '\t-----------------------------\n'

'''||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''
def return_x_seed():
	x_seed = np.zeros((1,SEQ_LENGTH,NUM_FEATURES))
	for q in xrange(SEQ_LENGTH/2):
		f_ind = 24 + np.random.choice(range(12))
		x_seed[0, 2*q, f_ind] = 1
		x_seed[0, 2*q+1, f_ind] = 1
	return x_seed


def generate_a_fugue(epoch, loss, N=16*32):
	'''
	INPUT:  epoch and loss: Passed just to name the output files.	
	N: How many steps the fugue is to be generated for. 
	'''
	'''	a = np.array([np.argmax(row) for row in pitch_matrix[:SEQ_LENGTH]])
	b = np.array([np.argmax(row) for row in x_seed_from_training[0]])
	print "ENTER: generate_a_fugue"
	print '[pitch, x_seed_from_training]:'
	print np.column_stack((a.T, b.T))'''
	# Advance the RNN model for N steps at its current state.
	fugue = np.zeros((N+SEQ_LENGTH, NUM_FEATURES))
	
	# x = return_x_seed()
	x = np.zeros_like(x_seed_from_training)
	x[:] = x_seed_from_training[:]

	fugue[:SEQ_LENGTH,:] = x[0,:,:]
	# print "epoch={}".format(epoch)
	for i in xrange(N):
		# Pick the note w/ highest probability.
		predict = probabilities(x) # predict.shape = (1, 61)
		ix = np.argmax(predict)
		
		fugue[i+SEQ_LENGTH][ix] = 1 
		x[0, :-1, :] = x[:, 1:, :] 
		x[0, -1, :] = fugue[i+SEQ_LENGTH, :] 
		
		# # # # # # # # # # # # # # # # # # 
	
	print_epoch = "%04d" % (epoch,)
	print_data_size = "%05d" % (data_size,)
	print_SEQ = "%02d" % (SEQ_LENGTH,)
	print_LSTM = "%01d" % (NUM_LAYERS,)
	
	filename =  'fugue_dsize' + print_data_size + \
					'_SEQ' + print_SEQ + \
					'_LSTM' + print_LSTM + \
					'_epoch' + print_epoch
					
	dump_dict = {'NUM_FEATURES' : NUM_FEATURES,
				'SEQ_LENGTH' : SEQ_LENGTH,
				'BATCH_SIZE' : BATCH_SIZE,
				'LEARNING_RATE' : LEARNING_RATE,
				'NUM_EPOCHS' : NUM_EPOCHS,
				'GRAD_CLIPPING' : GRAD_CLIPPING,
				'NUM_UNITS' : NUM_UNITS,
				'NUM_LAYERS' : NUM_LAYERS,
				'data_size' : data_size,
				'predict' : predict, 
				'loss' : loss,
				'fugue' : fugue}
				
	pickle.dump(dump_dict, open('Synthesized_Fugues/'+filename+'.p', 'wb'))
	time_series = pitch_matrix_TO_time_series_legato(fugue, sh=24)
	
	print 'len(time_series)={}'.format(len(time_series)), " "*100
	if len(time_series) > 0:
		time_series_TO_midi_file(time_series, 'Synthesized_Fugues/'+filename+'.mid')
	elif len(time_series) == 0:
		print 'ZERO NOTES PREDICTED'

	'''	a = np.array([np.argmax(row) for row in pitch_matrix[:SEQ_LENGTH]])
	b = np.array([np.argmax(row) for row in x_seed_from_training[0]])
	print "EXIT: generate_a_fugue"
	print '[pitch, x_seed_from_training]:'
	print np.column_stack((a.T, b.T))'''
	
	return
	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def build_rnn(sequence_length=SEQ_LENGTH, num_units=NUM_UNITS):
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
x_seed_from_training = np.zeros((1, SEQ_LENGTH, NUM_FEATURES))
x_seed_from_training[0] = pitch_matrix[:SEQ_LENGTH]

print "Building network ..."
l_in, l_out = build_rnn()
target_values = T.imatrix('target_output')
network_output = lasagne.layers.get_output(l_out)

# The loss function of predictions and target.
loss = T.mean(categorical_crossentropy(network_output, target_values))

# Retrieve all parameters from the network
all_params = get_all_params(l_out, trainable=True)

# Compute AdaGrad updates for training
print "Computing updates for training w/ adagrad..."
updates = adagrad(loss_or_grads=loss, params=all_params, learning_rate=LEARNING_RATE)

# Theano functions for training and computing loss
print "Compiling functions ..."
train_function = theano.function([l_in.input_var, target_values], loss, updates=updates, allow_input_downcast=True)
	
# Need probability distribution of the next note given 
# the state of the network and the input_seed.
# Compile a theano function called probabilities to produce 
# the probability distribution of the prediction.

# `probabilities` is analogous to .predict() function
probabilities = theano.function([l_in.input_var], network_output, allow_input_downcast=True)


'''  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || 
     \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/   
	   T  R  A  I  N  I  N  G  --  T  R  A  I  N  I  N  G  --  T  R  A  I  N  I  N  G												
'''
# make the training data tensors X and Y, and train the model on them
print "Training ..."
X, Y = make_X_Y(pitch_matrix, data_size, SEQ_LENGTH, NUM_FEATURES)
# print 'X.shape, Y.shape = {} ---- {}'.format(X.shape, Y.shape)

p, avg_loss, dummy_previous, loss_p = 0, 0, -1, []

for it in xrange(1, data_size * NUM_EPOCHS / BATCH_SIZE + 1):
	epoch = float(it) * float(BATCH_SIZE) / data_size
	print "it:", it, "--- epoch:", round(epoch,3), "     \r",
	time.sleep(1)
	x, y = make_batch(p, X, Y, data_size, BATCH_SIZE)
	avg_loss += train_function(x, y)
	
	if epoch % PRINT_FUGUES < dummy_previous:
		# generate a fugue every PRINT_FUGUES epochs
		generate_a_fugue(epoch=epoch, loss=loss_p) 
		t5 = time.time()
		print " "*100,"\r",
		print "Epoch {}, at {} sec, average loss={}".format(epoch, round(t5-t0), avg_loss)
	pass
	
	p = p + BATCH_SIZE
	dummy_previous = epoch % PRINT_FUGUES
	
	if p > data_size - 1:
		# Another epoch completed.
		# print "epoch {} completed. Reset: avg_loss=0, p=0".format(epoch)
		loss_p.append(avg_loss)
		p = 0
		avg_loss = 0
	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''

