from preprocess_2 import *
import midi
import re
import os
import matplotlib.pylab as plt

"""	Stage 0: MIDI to time_series_list conversion
"""
def plot_canonical_chords_vector():
	v, c = build_chords_vocabulary()
 	column_labels = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
	row_labels = [i.name for i in v]
	fig, ax = plt.subplots()
	heatmap = ax.pcolor(c, cmap=plt.cm.Blues)
	
	# put the major ticks at the middle of each cell
	ax.set_xticks(np.arange(c.shape[1])+0.5, minor=False)
	ax.set_yticks(np.arange(c.shape[0])+0.5, minor=False)
	
	ax.invert_yaxis()
	ax.xaxis.tick_top()
	ax.set_ylim([10,0])
	
	ax.set_xticklabels(column_labels, minor=False, fontsize=14)
	ax.set_yticklabels(row_labels, minor=False, fontsize=14)
	
	plt.ion()
	plt.show()
	


def plot_pitch_matrix(pitch_matrix, title=None, xlabel=None, ylabel=None):
	
	fig, ax = plt.subplots()
	heatmap = ax.pcolor(pitch_matrix, cmap=plt.cm.Blues)
	plt.title(title, fontsize = 20)
	plt.ylabel(ylabel, fontsize = 20)
	plt.xlabel(xlabel, fontsize = 20)
	ax = plt.gca()
	ax.set_xlim([0, pitch_matrix.shape[1]])
	ax.set_ylim([0, pitch_matrix.shape[0]])
	
	
	plt.ion()
	plt.show()
	
	
	
	
	
def extract_melody(track):
	'''
	INPUT: 	track MIDI.TRACK([]) OBJECT
	OUTPUT: note_on LIST [(tick INT, (pitch INT, velocity INT) TUPLE, time INT) TUPLE, ...],
			
			note_off LIST [(tick INT, (pitch INT, velocity INT) TUPLE, time INT) TUPLE, ...]
	'''
	note_on, note_off, time = [], [], 0
	for line in track:
				
		if line.name == 'Note On':
			time = time + line.tick
			tup = line.tick, line.data, time 
			if line.velocity == 0: # velocity=0. A note off event
				print 'velocity=0 @ NoteOnEvent()'
				note_off.append(tup)
			else:
				note_on.append(tup)
				
		elif line.name == 'Note Off':
			time = time + line.tick
			tup = line.tick, line.data, time
			note_off.append(tup)
			
	return note_on, note_off




def time_series_builder(note_on, note_off):
	'''
	INPUT: note_on LIST [(tick INT, (pitch INT, velocity INT), time INT), ...],
	 	   note_off LIST [(tick INT, (pitch INT, velocity INT), time INT), ...],
	OUTPUT: time_series LIST [(time INT, pitch INT, duration INT)]
	'''
	time_series = []
	ind = 0 # index at note_off
	for ev_on in note_on:
		pitch = ev_on[1][0]
		time = ev_on[2]
		for i, ev_off in enumerate(note_off):
			if pitch == ev_off[1][0]:
				duration = ev_off[2] - time
				note_off.pop(i)
				time_series.append((time, pitch, duration))
				if duration < 0: 
					print 'duration < 0\n\tev_on, ev_off', ev_on, ev_off
				break

	return time_series




def note_value(time_series_list):
	'''
	INPUT: time_series_list LIST
	OUTPUT: l LIST
	l is a sorted list of all possible note values in the midi file.
	96 ticks is a quarter note
	'''
	s = set() # possible note values
	[[s.add(ts[i][0]-ts[i-1][0]) for i in range(1, len(ts))] for ts in time_series_list]
	
	l = list(s)
	l.sort()
		
	return l



def extract_end_start_times(time_series_list, bar):

	start_time = min(time_series_list, key=lambda x: x[0])[0][0]
	# round it down to previous quarter note
	start_time = start_time - start_time % bar

	# take the duration of the note into acct while calculating end_time
	end_time = max([ts[-1][0] + ts[-1][2] for ts in time_series_list])
	# round it up to next quarter note
	end_time = end_time - end_time % bar + bar

	return start_time, end_time




def extract_slice(time_series_list, time, bar):
	'''
	INPUT: ..., INT, INT 
	       ..., tick count, tick count
	OUTPUT: LIST [(INT, INT, INT), ...] __ [(updated_time, pitch, clipped_duration), ...]
		clips the duration of the notes
		extract slice of all noteEvents with updated keyOn times, pitches and clipped durations
		Definitely and O(n) operation
	'''
	# select all noteEvents (rows of time_series_list) 
	# where keyOff > time

	dum = []
	for ts in time_series_list:
		for nE in ts:
			keyOn = nE[0]
			keyOff = nE[0]+nE[2]
			if keyOff > time:
				dum.append((keyOn if keyOn > time else time ,\
							 nE[1],\
							 nE[2] if keyOn > time else keyOff - time\
							))
					
	# intersection with all noteEvents 
	# where keyOn < time + bar 
	
	slice = []
	for nE in dum:
		keyOn = nE[0]
		keyOff = nE[0]+nE[2]
		if keyOn < time + bar:
			slice.append((keyOn,\
						 nE[1],\
						 nE[2] if keyOff < (time + bar) else (time + bar - keyOn)\
						))
	return slice




def extract_pitches(time_series_list, time=96*4*10, bar=96):
	'''
	INPUT: 
	OUTPUT: sth-by-2 NP.ARRAY(), np.array([np.array([ pitch INT, clipped_duration INT])])
	'''
	
	slice = extract_slice(time_series_list, time, bar)
	gr_pitches = np.empty((0,2), dtype=int)
	for nE in slice:
		gr_pitches = np.vstack((gr_pitches, np.array([nE[1], nE[2]]))) # pitch, clipped_duration
		
			
	return gr_pitches[:,0]




def extract_chord_sequence(time_series_list, bar=96):
    # INPUT : time_series_list [(time, pitch, duration)], 
    # OUTPUT: LIST of strings ['Am', 'GMaj aug', 'G7', 'Dm_harmonic', ...]
	
	start_time, end_time = extract_end_start_times(time_series_list, bar)
	
	time_sequence = range(start_time, end_time, bar)
	chord_sequence = [[None, None] for i in time_sequence] 

	for i, time in enumerate(time_sequence):
		gr_pitches = extract_pitches(time_series_list, time, bar)
		chord_sequence[i][0] = time
		chord_sequence[i][1] = find_chord(gr_pitches)
		print chord_sequence[i][1]

	return chord_sequence
		
def extract_pitch_matrix(time_series_list, bar=96):
	start_time, end_time = extract_end_start_times(time_series_list, bar)

	time_sequence = range(start_time, end_time, bar)
	pitch_matrix = np.zeros((128, len(time_sequence)), dtype=int)

	for i, time in enumerate(time_sequence):
		gr_pitches = extract_pitches(time_series_list, time, bar)
		for j in gr_pitches:
			pitch_matrix[j][i] = 1
		

	return pitch_matrix


if __name__ == '__main__':
	'''
	INPUT: filename STR 
	OUTPUT: time_series_list LIST [[ (time INT, pitch INT, duration INT) TUPLE, ...], ...]
	
	filename of MIDI file
	'''
	input_MIDI = 'bwv733_io.mid'
	tracks = midi.read_midifile(input_MIDI)
	note_on, note_off, time_series_list = \
			[0 for i in tracks], [0 for i in tracks], []
	
	for i, track in enumerate(tracks):
		note_on[i], note_off[i] = extract_melody(track)
		if len(note_on[i]) != len(note_off[i]):
			print '''len(note_on)={} and len(note_off)={} @track={}'''\
			.format(len(note_on[i]),len(note_off[i]),i)
		
		out = time_series_builder(note_on[i], note_off[i])
		if len (out) > 0:
			time_series_list.append(out)

	# # # # # # # 
	l = note_value(time_series_list)
	# # # # # # # 
	
	
	
	# # # # # # 
	# pitch_matrix = extract_pitch_matrix(time_series_list, bar=96)
	# plot_pitch_matrix(pitch_matrix, title='bwv733.mid pitch_matrix', xlabel = 'quarter note', ylabel='midi notes (0-127)')
	chord_sequence = extract_chord_sequence(time_series_list, bar=96*4)
	chord_sequence




