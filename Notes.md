							June 19, 2016 - Sunday

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
`time_series = 

--- 
**FINDING**

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



















