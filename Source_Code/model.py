import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, RecurrentLayer, LSTMLayer, DenseLayer, get_all_params
from lasagne.updates import adagrad
from lasagne.nonlinearities import softmax
from lasagne.objectives import binary_crossentropy, aggregate
import pickle
import time

t0 = time.time()
print "\nLoading pitch_matrix ..."
pitch_matrix = pickle.load(open('training_data/pitch_matrix_'+str(96/4)+'ticks_sh'+str(0)+'.p', 'rb'))
t1 = time.time()
print "\t\ttime: ", round(t1-t0, 4), "sec"

NUM_FEATURES = pitch_matrix[0].shape[0] # 73
SEQUENCE_LENGTH = 8
BATCH_SIZE = 48
LEARNING_RATE = .01
NUM_EPOCHS = 10
PRINT_FREQ = 1000

D = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


'''||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  
'''

def generate_a_fugue(N=96/4*16*16):
	# Advance the RNN model N times generating a note at each step. 
	# generate N many notes. 	
	fugue_new = np.zeros((N, NUM_FEATURES))
	
	x, _ = gen_data(p, pitch_matrix)
	print "generate_a_fugue,  p={}".format(p)
	for i in range(N):
		# Pick only a single note that got assigned the highest probability.
		# Will be modified to accommodate polyphonic music creation
		ix = np.argmax(probs(x).ravel())
		
		# # # # # # # # # # # # # # # # # # 
		fugue_new[i][ix] = 1 
		x[:, 0:SEQUENCE_LENGTH-1, :] = x[:, 1:, :] 
		x[:, SEQUENCE_LENGTH-1, :] = 0 
		x[0, SEQUENCE_LENGTH-1, :] = fugue_new[i, :] 

	phrase = ''
	for ix in fugue_new:
		for i,_ in enumerate(ix):
			if _ == 1: break
		note = D[i%12]
		octave = str(i/12)
		phrase = '{}  {}-{}  '.format(phrase, note, octave) 
		
	random_snippet = ''.join(phrase) 
	print("----\n %s \n----" % random_snippet)


def input_seed(pitch_matrix):
	# starting from the middle index of the pitch matrix
	# extract a segment of length SEQUENCE_LENGTH
	# then strecth the notes to 2*SEQUENCE_LENGTH
	# eg. A, C#, D => A, A, C#, C#, C#, C#, D, D
	
	
	mid = data_size/2
	extract = pitch_matrix[mid:mid+SEQUENCE_LENGTH]
	
	generation_phrase = np.zeros((SEQUENCE_LENGTH*2,pitch_matrix.shape[1]))
	generation_phrase[range(0,SEQUENCE_LENGTH*2,2)] = extract
	generation_phrase[range(1,SEQUENCE_LENGTH*2+1,2)] = extract
	
	return generation_phrase


def gen_data(p=0, pitch_matrix=pitch_matrix, batch_size=BATCH_SIZE):
	x = np.zeros((batch_size, SEQUENCE_LENGTH, NUM_FEATURES))
	y = np.zeros((batch_size, NUM_FEATURES))
	for n in range(batch_size):
		x[n, : , :] = pitch_matrix[p+n:p+n+SEQUENCE_LENGTH, :]
		y[n, :] = pitch_matrix[p+n+SEQUENCE_LENGTH]
		return x, y
		
def flatten_pitch_matrix(pitch_matrix):
	'''
	INPUT: list of 2d np.arrays 
	OUTPUT: 2d np.array 
	'''
	# flatten pitch_matrix
	pitch_matrix_long = pitch_matrix[0]
	for n in range(len(pitch_matrix)-1):
		pitch_matrix_long = np.column_stack((pitch_matrix_long, pitch_matrix[n+1]))

	# transpose it to swap the axes. 
	# pitch_matrix.shape = (N, NUM_FEATURES)
	pitch_matrix = np.transpose(pitch_matrix_long)	
	return pitch_matrix 

''' ................................ '''

def build_rnn(sequence_length=SEQUENCE_LENGTH, num_units=512):
	# input layer
	# generate the data to pass into this
	l_in = InputLayer(shape=(None, sequence_length, NUM_FEATURES))
	# LSTM Layer
	l_LSTM = LSTMLayer(l_in, num_units=num_units, grad_clipping=100, only_return_final=True)
	# output layer
	l_out = DenseLayer(l_LSTM, num_units=NUM_FEATURES, nonlinearity=softmax)
	return l_in, l_out


def total_cost(predictions, target):
	'''
	INPUT np.array of size (73,), np.array of size (73,)
	OUTPUT: 
	'''
	note_cost = binary_crossentropy(predictions, target).mean()
	
	# chord_target = get_chord(target)
	# note_predictions = predictions[predictions > .5]
	# chord_predictions = get_chord(note_predictions)
	# chord_cost = cosine_distance(chord_predictions, chord_target)
	
	chord_cost = 0
	return note_cost + chord_cost
	
	
'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  
'''

pitch_matrix = flatten_pitch_matrix(pitch_matrix)
print "(Retain only first 9600 notes in pitch_matrix)"
pitch_matrix = pitch_matrix[0:9600]
data_size = pitch_matrix.shape[0]

print "Building network ..."
l_in, l_out = build_rnn()
target_values = T.imatrix('target_output')
network_output = lasagne.layers.get_output(l_out)

# The loss function of predictions and target.
cost = total_cost(network_output, target_values)

# Retrieve all parameters from the network
all_params = get_all_params(l_out, trainable=True)

# Compute AdaGrad updates for training
print "Computing updates w/ adagrad..."
updates = adagrad(loss_or_grads=cost, params=all_params, learning_rate=LEARNING_RATE)
t2 = time.time()
print "\t\ttime: ", round(t2-t0, 4), "sec"

# Theano functions for training and computing cost
print "Compiling functions ..."
train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
compute_cost = theano.function([l_in.input_var, target_values], cost, allow_input_downcast=True)
	
# Need probability distribution of the next notes given 
# the state of the network and the input_seed.
# Compile a theano function called probs to produce 
# the probability distribution of the prediction.

probs = theano.function([l_in.input_var], network_output, allow_input_downcast=True)

# The next function generates text given a phrase of length at least of SEQUENCE_LENGTH.
# The phrase is set using the variable input_seed.
# The optional input "N" is used to set the number of characters of text to predict. 
t3 = time.time()
print "\t\ttime: ", round(t3-t0, 4), "sec"

generation_phrase = input_seed(pitch_matrix)

	
''' ................................ '''	
''' ................................ '''

print "Training ..."
p = 0
for it in xrange(data_size * NUM_EPOCHS / BATCH_SIZE):
	print "it: {} -- data_size = {}".format(it, data_size)
	t4 = time.time()
	# Generate a fugue starting from a SEQUENCE_LENGTH long segment starting at the p-th note of pitch_matrix
	generate_a_fugue() 
	t5 = time.time()
	print "\t\ttime: ", round(t5-t4, 4), "sec"
	print "\t\tTOTAL TIME: ", round(t5-t0, 4), "sec"
	
	avg_cost = 0;
	for _ in range(PRINT_FREQ):
		x,y = gen_data(p, pitch_matrix)
		
		
		p += SEQUENCE_LENGTH + BATCH_SIZE - 1 
		if(p+BATCH_SIZE+SEQUENCE_LENGTH >= data_size):
			# print('Carriage Return')
			p = 0;
			
		avg_cost += train(x, y)
	print("Epoch {} average loss = {}".format(it*1.0*PRINT_FREQ/data_size*BATCH_SIZE, avg_cost / PRINT_FREQ))
	

'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || 
'''
	
	
	
	
	
	
	
	
	
	
