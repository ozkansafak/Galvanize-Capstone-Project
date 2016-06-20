import midi
import re

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
	data = [int(elem) for elem in  re.findall(r'\D*(\d*)\D+', data_p)]
	return tick, data, time+tick
	
def extract_melody(track):
	'''
	INPUT: 	track MIDI.TRACK([...]) OBJECT
	OUTPUT: note_on LIST [(tick INT, (pitch INT, velocity INT), time INT)],
			
			note_off LIST [(tick INT, (pitch INT, velocity INT), time INT)]
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
	INPUT: note_on LIST [tick INT, (pitch INT, velocity INT)],
	 	   note_off LIST [tick INT, (pitch INT, velocity INT)],
	OUTPUT: note LIST [note INT, duration INT]
	'''
	time_series = []
	ind = 0 # index at note_off
	for ev1 in note_on:
		pitch = ev1[1][0]
		time = ev1[2]
		for ev2 in note_off:
			if pitch == ev2[1][0]:
				duration = ev2[2] - time
				if duration < 0:
					print 'duration < 0'
					print 'ev1, ev2', ev1, ev2
				note_off.pop(i)
		time_series.append((time, pitch, duration))
		
	return time_series		

if __name__ == '__main__':
	input_MIDI = './Python MIDI/bwv733.mid'
	tracks = midi.read_midifile(input_MIDI)
	note_on, note_off = [0 for i in tracks], [0 for i in tracks]
	
	for i, track in enumerate(tracks):
		note_on[i], note_off[i] = extract_melody(track)
		if len(note_on[i]) != len(note_off[i]):
			print 'note_on and note_off are of different lengths'
		
		time_series = time_series_builder(note_on[i], note_off[i])
		







