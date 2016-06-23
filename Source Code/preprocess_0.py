from preprocess_2 import *
import midi
import re
import os

"""	Stage 0: MIDI to time_series_list conversion
"""

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
				if duration < 0: print 'duration < 0\n\tev_on, ev_off', ev_on, ev_off
				break

	return time_series

def note_value(time_series_list):
	'''
	INPUT: time_series_list LIST
	OUTPUT: l LIST
	l is a sorted list of all possible note values in the midi file.
	'''
	s = set() # possible note values
	[[s.add(ts[i][0]-ts[i-1][0]) for i in range(1, len(ts))] for ts in time_series_list]
	
	l = list(s)
	l.sort()
	
	# for i in range(1, len(l)):
	# 	ratio = l[i]/96. # 96 ticks to a quarter note
	# 	print 'note_values in current file: {}'.format(ratio)
	
	return l


def extract_slice(time_series_list, time=96*4*10, bar=96):
	
	dum = []
	for time_series in time_series_list:
		dum.append([elem for elem in time_series if elem[0] >= time])

	
	slice =[]
	for time_series in dum:
		slice.append([elem for elem in time_series if elem[0] < time + bar])
	
	return slice


if __name__ == '__main__':
	"""
	INPUT: filename STR 
	OUTPUT: time_series_list LIST [[ (time INT, pitch INT, duration INT) TUPLE, ...], ...]
	
	filename of MIDI file
	"""
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
	
	slice = extract_slice(time_series_list)
	
	gr_pitches = np.empty(0, dtype=int)
	for ts in slice:
		for elem in ts:
			gr_pitches = np.append(gr_pitches, elem[1])
	
	
	root, chord_type = find_chord(gr_pitches)
	print root, chord_type
		




