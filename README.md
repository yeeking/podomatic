# podomatic
Automatic podcast editing system written in Python

Takes two video files, analyses the audio and automatically makes an edit list which selects the visible clip based on which is the loudest mic 

```
mkdir -p ~/Python/podomatic-env
python3 -m venv ~/Python/podomatic-env
source ~/Python/podomatic-env/bin/activate
pip install numpy scipy moviepy librosa matplotlib pedalboard ffmpeg-python 
```

```
# for now I have hardcoded the filenames in the script but eventually it'll be...
python src/podomatic-v1.py infile1.mov infile2.mov outfile.mov clean-audio-out.wav processed-audio-out.wav
```
You should definitely make sure the two videos are in sync before running it. 


