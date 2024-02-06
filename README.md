# podomatic
Automatic podcast editing system written in Python

Takes two video files, analyses the audio and automatically generates an edited video which switches to the camera with the sound. It also mixes the two audio tracks and applies a naive gating and limiting process. The mixed audio is also written out in clean and processed form. 

```
# If you want to make a virtualenv for it: 
mkdir -p ~/Python/podomatic-env
python3 -m venv ~/Python/podomatic-env
source ~/Python/podomatic-env/bin/activate
# required packages
pip install numpy scipy moviepy librosa matplotlib pedalboard ffmpeg-python 
```

Now prepare your two input video files so they are reasonably in-sync, then run this command 

```
# args: input video 1, input video 2, output video, output_audio_processed, output_audio_clean
# the file extension on outfile dictates the codecs: mp4 -> h264/aac or mov -> prores/s24le
python src/podomatic-v1.py infile1.mov infile2.mov outfile.mov processed-audio-out.wav clean-audio-out.wav
```
 


