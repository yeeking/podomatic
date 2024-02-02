# podomatic
Automatic podcast editing system written in Python

Takes two video files, analyses the audio and automatically makes an edit list which selects the visible clip based on which is the loudest mic 

```
python podomatic.py infile1.mov infile2.mov outfile.mov 
```
You should probably make sure the two videos are in sync before running it. 

