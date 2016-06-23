***June 19, 2016 - Sunday***

Believe it or not, I accidentally ran `rm -rf /*` on my computer when i wanted to type  `rm -rf ./*`!!!.

I stopped the deletion  when it was going thru `Applications/Evernote.app` directory. It was throwing `Permission denied` errors all over the place. But, it deleted Ableton Live and all my instrument patches. I ended up having to download Ableton from their website. It was easy because I registered it when I bought it. I'll have to re-install the patches some other time. I hope this is the whole extent of the damage.

---
Deleting .DS_Store remotely on git repo:
```javascript
find . -name ".DS_Store" -exec git rm --cached -f {} \;.
git commit -m "delete files"
git push
```
---
**Python MIDI**

https://github.com/vishnubob/python-midi

Python, for all its amazing ability out of the box, does not provide you with an easy means to manipulate MIDI data. There are probably about ten different python packages out there that accomplish some part of this goal, but there is nothing that is totally comprehensive.

---
**md file cheatsheet**

https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet

---
**Andrej Karpathy RNN library**

https://github.com/karpathy/char-rnn

---
**MIDI PYTHON**
Midi keyboard offers 128 keys (0 thru 127), whereas a piano offers 88 keys.

```javascript
midi.C_0 => returns 0
midi.Cs_0 => returns 1  it's C sharp
midi.Db_0 => returns 1  it's D flat
midi.D_0 => returns 2
midi.C_1 => returns 12 (one octave above midi.C_0)
.
.
.
midi.G_10 => returns 127 (highest note on MIDI)
```
- - - - - - 

```javascript
midi.NoteOnEvent(tick=24, channel=0, data=[62, 127]),
```

In a MIDI file, all NoteOn and NoteOff events are ordered sequentially. 
* The `tick` argument is set to tick count after the most recent NoteOn or NoteOff Event. 
* `data` argument is equal to `[pitch, velocity]` of the current note. 
* `velocity` is how hard the note is played. Varies from `0` to `127`.
* If the `velocity` is set to `0` in a `NoteOnEvent()`, it becomes a `NoteOffEvent()`. 

---
`Source Code/Python MIDI`
preprocessing Stage 0 finished
`OUTPUT: time_series LIST = [(time, pitch, duration), (...), (...)]`

--- 
**IMPORTANT FINDING !!!**

I have only worked with one single Bach Fugue MIDI File: bwv733.mid
It looks like there were errors in this midi file. And there might be similar errors on other data files.
```
at track=1: len(NoteOnEvent) = 849
            len(NoteOffEvent) = 850
```
Also, my `preprocessing.py` script gave negative `duration` error. 
However, when I imported the midi file to **Ableton Live** and then exported it back and then ran the `preprocessing.py` on it, all the negative `duration` errors were gone!!! and the redundant NoteOff events were also gone. 
```
at track=1: len(NoteOnEvent) = 849
            len(NoteOffEvent) = 849
```

I might need to manually import and export the midi files in and out of Ableton Live before feeding them to RNN as training set. 

Tomorrow:  
> Tests to check basic parameters.  
> Import the time_series back to MIDI 

---

***June 20, 2016 - Monday***

Had a great meeting/ Brainstorming session with Kamil.  
**A Crude Road Map:**
``` 
- Stage 0: Key & Chord Sequence Prediction
- Stage 1: INPUT: Bass line
           OUTPUT: Generate a Melody
- Stage 2: INPUT: Ignore the Bass Line and consider the new Melody created
           OUTPUT: Write a new Melody.
```


**pre-processing Helper Functions.**
```
- Stage 0: MIDI to TimeSeries -- finished
- Stage 1: TimeSeries to MIDI -- today
- Stage 2: TimeSeries to ChordSequence
- Stage 3: ChordSequence to Key
```
**Interaction btw Bass Lines and Melody Lines**
```
..................................(t-1).t............
Melody: 1---2---2---1---5---5---5--|5|-|4|--6---6---4

..Bass: 1---3---3---3---2---2---2---2--|4|--4---4---4
```

Set up the RNN such that the melody note at `time = t` is influenced by a melody note at `time = t-1` and bass note at `time = t`

---

***June 21, 2016 - Tuesday***


**Preprocess Pipe Line**
```
          (import)      (export) |bwv733_t1.mid| (merger)
bwv733.mid ----> Ableton ------> |bwv733_t2.mid| -------> bwv733_io.mid
                                 |bwv733_t3.mid|
```
When `bwv733.mid` is imported into and exported out of Ableton the timeSignature info in the midi file gets modified. I tried to reimport it from the original `bwv733.mid` file to no avail. 
I'll abandon this endeavor now and instead will write a script to get the note value info based on all the tracks in the midi file. 

**chords**
```javascript
canonical_chord_vectors
[0.46  0.0  0.07 0.35 0.0   0.14 0.0   0.35  0.07  0.0    0.14   0.0  ] minor
[0.46  0.0  0.07 0.35 0.0   0.14 0.0   0.35  0.07  0.0    0.     0.14 ] minor_harmonic
[0.45  0.0  0.07 0.34 0.0   0.14 0.0   0.34  0.0   0.203  0.     0.203] minor_melodic
[0.46  0.0  0.07 0.35 0.0   0.14 0.35  0.0   0.07  0.14   0.     0.035] minor_diminished
[0.46  0.0  0.07 0.35 0.0   0.14 0.35  0.0   0.07  0.0    0.14   0.0  ] minor_half_diminished
[0.38  0.0  0.06 0.0  0.29  0.11 0.0   0.29  0.0   0.057  0.     0.114] major
[0.38  0.0  0.06 0.0  0.29  0.0  0.12  0.12  0.0   0.058  0.     0.116] major_augmented
[0.38  0.0  0.06 0.0  0.29  0.12 0.0   0.29  0.0   0.058 -0.576  0.0  ] dominant
[0.37  0.11 0.0  0.11 0.28  0.0  0.29  0.0   0.113 0.0    0.113  0.0  ] dominant_altered
[0.33  0.0  0.03 0.0  0.26  0.0  0.15  0.25  0.0   0.025  0.099  0.0  ] dominant_sharp_11

```

---

***June 22, 2016 - Wednesday***

Working on chord-note similarity.  
Today's to do list:

*  Run the Chord Sequencer on a MIDI file. Calculate the sequence of chords per quarter note. 
```
def extract_chord_sequence(filename):
	# INPUT : filename STR, 
	# OUTPUT: LIST of strings ['Am', 'GMaj aug', 'G7', 'Dm_harmonic', ...]
```
*  Create RNN Input file for each MIDI (Lee)
*  Implement the Cost Function term that accounts for chord-melody mismath (Lee)

```python
# group of pitches. This is in Cm [0,3,7,10] transposed up 3 semitones.
In [5]: gr_pitches = np.array([7+12+3, 10+24+3, 7+24+3, 0+12+3, 3+24+3, 3+24+3, 0+36+3])

In [6]: gr_notes = pitches_to_notes(gr_pitches); gr_notes
Out[6]: array([ 1,  3,  6, 10])

In [7]: find_chord(gr_pitches)
Out[7]: ('D#', 'minor')
```

---

`pitch_matrix` is sucessfully constructed on **quarter notes**


*12:10 AM Bart, sitting on the ground against the wall*

I was able to run the code that builds 
`chord_sequence`. The chords are not looking so bad, either. 
Thus far, I made it only for `bwv733.md`. 

Note: When I have time I will think of a better way to construct 
`canonical_chord_vectors`. I'm lacking a foolproof consistency across 
the vocab of chords. The Root, 3rd, 5th and 7th should all have same 
weights across the board including `dominant altered` and
`minor harmonic octatonic scale`. Then, I have to find a way
to deal with color tones for the more complicated chord. It's doable. Only not my top priority righ now.

Tomorrow:  

* Time to go thru all `bwv*.mid` files to prepare the training data. 
Use `os` module to take the filenames in and construct a `pitch_matrix` 
and a `chord_sequence` for all of them. Then I can `cPickle` the output and feed it as training data to RNN model. 
* build your custom RNN model on Lasagne.

![pitchmatrix](Source-Code/pitch_matrix.png)


---


***June 23, 2016 - Thursday***



