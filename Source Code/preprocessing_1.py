import midi
import re
import os

"""	Stage 1: MIDI to time_series conversion
"""

def time_series_to_MIDI_track(time_series):
	'''
	INPUT: time_series LIST [(time, pitch, duration), (...), ...]
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
		pattern.append(track)
		# Instantiate a MIDI note on event, append it to the track
	
	
	# Add end of track event
	eot = midi.EndOfTrackEvent(tick=1)
	track.append(eot)
	# Print out the pattern
	print pattern
	# Save the pattern to disk
	midi.write_midifile("example.mid", pattern)








