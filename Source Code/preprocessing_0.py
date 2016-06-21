import midi
import re
import os

"""	Stage 0: MIDI to time_series_list conversion
"""

def single_note_extractor(m, time):
	"""
	INPUT: 	m, REGEX.MATCH OBJECT
	OUTPUT: (tick INT, data (INT, INT), time INT)
			
			tick is the tick count since last NoteOn/NoteOff event
			Need to calculate time since start of midi file.
	"""
	
	tick_p = m.group(1)
	tick = int(re.findall(r'\d+', tick_p)[0])
	data_p = m.group(3)
	data = tuple(int(elem) for elem in  re.findall(r'\D*(\d*)\D+', data_p))
	return tick, data, time + tick
	
def extract_melody(track):
	'''
	INPUT: 	track MIDI.TRACK([...]) OBJECT
	OUTPUT: note_on LIST [(tick INT, (pitch INT, velocity INT), time INT), ...],
			
			note_off LIST [(tick INT, (pitch INT, velocity INT), time INT), ...]
	'''
	note_on, note_off, time = [], [], 0
	for line in track:
		line = str(line)
		
		r_on = r'.*midi\.NoteOnEvent.*tick=(\d*).*channel=(\d*).*data=(.*)'
		r_off = r'.*midi\.NoteOffEvent.*tick=(\d*).*channel=(\d*).*data=(.*)'
		m_note_on = re.match(r_on, line)
		m_note_off = re.match(r_off, line)
		
		if m_note_on:
			tup = single_note_extractor(m_note_on, time)
			time = tup[2]
			if tup[1][1] == 0: # velocity=0. A note off event
				print 'velocity=0 @ NoteOnEvent()'
				note_off.append(tup)
			else:
				note_on.append(tup)
		elif m_note_off:
			tup = single_note_extractor(m_note_off, time)
			time = tup[2]
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

if __name__ == '__main__':
	"""
	INPUT: MIDI file
	OUTPUT: time_series_list
	"""
	#ls = os.listdir('../MIDI files/jsbach.net')
	input_MIDI = 'bwv733.mid'
	tracks = midi.read_midifile(input_MIDI)
	note_on, note_off, time_series_list = \
			[0 for i in tracks], [0 for i in tracks], [0 for i in tracks]
	
	for i, track in enumerate(tracks):
		note_on[i], note_off[i] = extract_melody(track)
		if len(note_on[i]) != len(note_off[i]):
			print '''len(note_on)={} and len(note_off)={} @track={}'''\
			.format(len(note_on[i]),len(note_off[i]),i)
		
		time_series_list[i] = time_series_builder(note_on[i], note_off[i])
		







