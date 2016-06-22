*June 19, 2016 - Sunday*

Believe it or not, I accidentally ran `rm -rf /*` on my computer. and left the `.` out of the `./*` part.

I stopped it when it was going thru `Applications/Evernote.app` directory. It was throwing `Permission denied` error all over the place. But, it deleted Ableton Live and all my instrument patches. I downloaded Ableton from their website. It was easy because I'm a registered customer. And I'll re-install the patches some other time. I hope this is the whole extent of the damage.

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
**IMPORTANT FINDING**

I have only worked with one single Bach Fugue MIDI File: bwv733.mid
It looks like there were errors in this midi file. And there might be similar errors on other data files.
```
at track=1: len(NoteOnEvent) = 849
            len(NoteOffEvent) = 850
```
Also, my `preprocessing.py` script gave negative `duration` error. However, when I imported the midi file to **Ableton Live** and then exported it back and then ran the `preprocessing.py` on it, all the negative `duration` errors were gone!!! and the redundant NOteOff events were also gone. 
```
at track=1: len(NoteOnEvent) = 849
            len(NoteOffEvent) = 849
```

> I might need to manually import and export the midi files in and out of Ableton Live before feeding them to RNN as training set. 

Tomorrow: 
- do more tests to check basic parameters. 
- import the time_series back to MIDI 

---

*June 20, 2016 - Monday*

Had a great meeting/ Brainstorming session with Kamil.
**A crude Road Map:**
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

*June 21, 2016 - Tuesday*


**Pipe Line**
```
          (import)      (export) |bwv733_t1.mid| (merger)
bwv733.mid ----> Ableton ------> |bwv733_t2.mid| -------> bwv733_io.mid
                                 |bwv733_t3.mid|
```
when `bwv733.mid` is imported into and exported out of Ableton the timeSignature info in the midi file gets modified. I tried to reimport it from the oroginal `bwv733.mid` file to no avail. 
I'll abandon this endeavor now and instead will write a script to get the note value info based on all the tracks in the midi file. 

**chords**
```
0 [ 1.   -0.1   0.1   0.75 -1.    0.3  -0.1   0.75  0.1  -0.1   0.3  -0.1 ] minor
1 [ 1.   -0.1   0.1   0.75 -1.    0.3  -0.1   0.75  0.1  -0.1  -0.1   0.3 ] minor_harmonic
2 [ 1.    0.1  -0.1   0.75 -1.   -0.1   0.75  0.1  -0.1   0.3  -0.1  -0.1 ] minor_diminished
3 [ 1.   -0.1   0.1   0.75 -1.    0.3   0.75 -0.1   0.1  -0.1   0.3  -0.1 ] minor_half_diminished
4 [ 1.   -0.1   0.1  -1.    0.75  0.3  -0.1   0.75 -0.1   0.1  -0.1   0.3 ] major
5 [ 1.   -0.1   0.1  -1.    0.75 -0.1   0.3   0.75 -0.1   0.1  -0.1   0.3 ] major_augmented
6 [ 1.   -0.1   0.1  -1.    0.75  0.3  -0.1   0.75 -0.1   0.1   0.3  -0.1 ] dominant
7 [ 1.    0.1  -0.1   0.75  0.3  -0.1   0.75 -0.1   0.1  -0.1   0.3  -0.1 ] dominant_altered
8 [ 1.   -0.1   0.1  -0.1   0.75 -0.1   0.3   0.75 -0.1   0.1   0.3  -0.1 ] dominant_sharp_11
```






