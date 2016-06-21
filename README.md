### An Artificial Fugue Generator using RNNs

**A crude road Map of this project:**
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
 `Melody: 1  2  2  1  5  5  5|5||4|6  6  4`

 `..Bass: 3  3  3  3  2  2  2  2 |4|4  4  4`

Set up the RNN such that the melody note at `time = t` is influenced by a melody note at `time = t-1` and bass note at `time = t`



