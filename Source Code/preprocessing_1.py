import midi
import re
import os

"""	Stage 1: MIDI to time_series conversion
"""

def build_single_track(source, pattern):
	'''
	INPUT: track and pattern
	OUTPUT: pattern
	
			Break down the track to its note events. 
			Append the track to pattern and return pattern
	'''
	track = midi.Track()
	pattern.append(track)
	for line in source[0]:
	#	if (line.name == 'Note On') | (line.name == 'Note Off') | (line.name == 'End of Track'):
			track.append(line)

	return pattern
	
def midi_track_merger():
	'''
	INPUT: rootfilename STR (eg 'bwv733.mid')
	
		   merges the tracks found under, for instance,
		   'bwv733_t1.mid', 'bwv733_t2.mid' .. etc
		   into 'bwv733.mid'
	'''
	rootfilename = 'bwv733.mid'
	rootfilename = rootfilename[:-4]
	ls = os.listdir('.')
	lst = []
	for item in ls:
		if (item[0:len(rootfilename)] == rootfilename) and len(item) > len(rootfilename)+4:
			lst.append(item)
	# eg, lst = ['bwv733_t1.mid', 'bwv733_t2.mid', 'bwv733_t3.mid']
	
	# initialize pattern object
	pattern = midi.Pattern()
	for item in lst:
		source = midi.read_midifile(item)
		pattern = build_single_track(source, pattern)
		
	print pattern
	midi.write_midifile(rootfilename + '.mid', pattern)

def time_series_to_MIDI_track(time_series):
	'''
	INPUT: time_series LIST [(time, pitch, duration), ...]
	OUTPUT: track midi.Track()
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
	
	return track


if __name__=='__main__':
	# Instantiate a MIDI Pattern (contains a list of tracks)
	pattern = midi.Pattern()
	for time_series in time_series_list:
		track = time_series_to_MIDI_track(time_series)
		# Add end of track event
		eot = midi.EndOfTrackEvent(tick=1)
		track.append(eot)
		# append to pattern
		pattern.append(track)

	
	print pattern
	midi.write_midifile("track.mid", pattern)








