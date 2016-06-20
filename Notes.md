

Deleting .DS_Store from remotely on git repo:
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
* If the velocity is set to 0 in a NoteOnEvent(), it is a NoteOffEvent(). 
























