import numpy as np	
import midi


def adjust_pitch_matrix(pitch_matrix, data_size, batch_size, num_features):
	p = (data_size / batch_size) * batch_size
	leftOver = (p+batch_size) % (data_size)
	if leftOver > 0:
		print "adjust_pitch_matrix(): Broken batch. leftOver =", leftOver
		pitch_matrix = np.concatenate((pitch_matrix,np.zeros((leftOver,num_features))), axis=0)
		print 'adjust_pitch_matrix reports: pitch_matrix.shape =', pitch_matrix.shape

	return pitch_matrix, data_size + leftOver

def make_X_Y(pitch_matrix, data_size, sequence_length, num_features):

	X = np.zeros((data_size, sequence_length, num_features))
	Y = np.zeros((data_size, num_features))
	
	for i in range(data_size):
		X[i, : , :] = pitch_matrix[i : i+sequence_length, :]
		Y[i, :] = pitch_matrix[i+sequence_length]

	return X, Y
	

def make_batch(p, X, Y, data_size, batch_size):
	if p+batch_size-1 <= data_size-1:
		x = X[p : p+batch_size]
		y = Y[p : p+batch_size]
	else:
		# p = 12; batch_size = 6; data_size = 16
		# next batch: [12,13,14,15,16,17]
		# leftOver = 2
		leftOver = (p+batch_size) % (data_size)
		print "\nmake_batch(): Broken batch. leftOver=", leftOver

		x = X[p:]
		y = Y[p:]
		# reset p
		x1 = X[:leftOver]
		y1 = Y[:leftOver]
		#
		x = np.concatenate((x,x1), axis=0)
		y = np.concatenate((y,y1), axis=0)
		#
	return x, y

'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

def flatten_pitch_matrix(pitch_matrix):
	'''
	INPUT: list of 2d np.arrays 
	OUTPUT: 2d np.array 
	
	# patch the individual pitch_matrices of individual fugues
	# into a single 2D numpy array.
	# Transpose it, then return it.
	
	'''
	# flatten pitch_matrix
	pitch_matrix_long = pitch_matrix[0]
	for n in range(len(pitch_matrix)-1):
		pitch_matrix_long = np.column_stack((pitch_matrix_long, pitch_matrix[n+1]))

	# transpose it to swap the axes. 
	# pitch_matrix.shape = (N, NUM_FEATURES)
	pitch_matrix = np.transpose(pitch_matrix_long)	
	return pitch_matrix

'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''

def make_monophonic(pitch_matrix):
	'''
	This is a preprocess step that's invoked in model.py on AWS
	'''
	# pitch_matrix is already flattened upon being loaded in model.py
	# 		pitch_matrix.shape[1] = 73
	c=0
	for vec in pitch_matrix:
		c+=1
		for i, key in enumerate(vec):
			if key == 1: 
				break
		vec[i+1:] = 0
	
	return pitch_matrix

'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''

def pitch_matrix_TO_time_series_legato(pitch_matrix, bar=96/4, sh=0):
	
	""" This seems to work right """
	time_series = []
	for pitch, row in enumerate(pitch_matrix):
		i = 0	
		while i < len(row):
			if row[i] == 1:
				c = 0
				while i < len(row) and row[i] == 1:
					c+=1
					i+=1
				time_series.append(((i-c)*bar, pitch+sh, c*bar))
				i -= 1
			i += 1
	
	time_series.sort(key=lambda x: x[0])
	return time_series	

'''/\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\  /\ 
   ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  || '''

def time_series_TO_midi_file(time_series, filename="Synthesized_Fugues/out.mid"):
	'''
	INPUT: time_series LIST [(time, pitch, duration), ...]
	OUTPUT: track midi.Track()
	
	This is going to be used to convert RNN generated fugues back to MIDI
	'''
	track = midi.Track()
	
	# U list of noteEvents [('on', time, pitch), ...]
	U = []
	for (time, pitch, duration) in time_series:
		U.append(('on', time, pitch))
		U.append(('off', time+duration, pitch))
	
	#order the list of events	
	U.sort(key=lambda x: x[1])
	cursor = 0
	for (typ, time, pitch) in U:
		tick = time - cursor
		if typ == 'on':
			o = midi.NoteOnEvent(tick=tick, velocity=127, pitch=pitch)
		else:
			o = midi.NoteOffEvent(tick=tick, velocity=127, pitch=pitch)
		track.append(o)
		cursor = time

	# Add end of track event
	eot = midi.EndOfTrackEvent(tick=1)
	track.append(eot)
	
	pattern = midi.Pattern()
	pattern.append(track)
	midi.write_midifile(filename, pattern)
	
	return


'''||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/ '''



if __name__ == '__main__':
	# time_series_list = time_series_list_builder(filepath)
	pitch_matrix = time_series_list_TO_pitch_matrix(time_series_list)
	time_series = pitch_matrix_TO_time_series_legato(pitch_matrix)
	time_series_TO_midi_file(time_series)
	







