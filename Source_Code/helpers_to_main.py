import numpy as np	
import midi
D = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


'''||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||  ||
   \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  \/  '''

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
	







