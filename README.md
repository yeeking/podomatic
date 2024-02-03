# podomatic
Automatic podcast editing system written in Python

Takes two video files, analyses the audio and automatically makes an edit list which selects the visible clip based on which is the loudest mic 

```
pip install numpy scipy moviepy librosa matplotlib pedalboard fmpeg-python 
```

```
# for now I have hardcoded the filenames in the script but eventually it'll be...
python podomatic-v1.py infile1.mov infile2.mov outfile.mov clean-audio-out.wav processed-audio-out.wav
```
You should definitely make sure the two videos are in sync before running it. 


