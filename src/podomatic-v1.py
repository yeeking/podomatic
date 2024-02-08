import tempfile
import numpy as np
import librosa
from moviepy.editor import VideoFileClip, concatenate_videoclips
import scipy 
from pedalboard import Pedalboard, Chorus, Reverb, NoiseGate
from pedalboard.io import AudioFile
import ffmpeg
import os
from scipy.io import wavfile
from moviepy.editor import AudioFileClip, AudioClip
import os
from pedalboard import Pedalboard, Chorus, Reverb, Limiter
import platform


def get_audio_sample_rate(video_file):
    probe = ffmpeg.probe(video_file, v='quiet', select_streams='a:0', show_entries='stream=sample_rate')
    if 'streams' in probe and len(probe['streams']) > 0:
        return int(probe['streams'][0]['sample_rate'])
    else:
        return 0  # No audio or first audio track not found

def count_audio_channels(video_file):
    probe = ffmpeg.probe(video_file, v='quiet', select_streams='a:0', show_entries='stream=channels')
    if 'streams' in probe and len(probe['streams']) > 0:
        return int(probe['streams'][0]['channels'])
    else:
        return 0  # No audio or first audio track not found

def extract_audio_data(video_file):
    """
    extracts the audio track from the video file and returns it as an
    array of samples. If 2 channels, returns average values for simplicity 
    """
    sr = get_audio_sample_rate(video_file)
    channels = count_audio_channels(video_file)
    audio, _ = (
        ffmpeg.input(video_file)
        .output("pipe:", format="f32le", acodec="pcm_f32le")
        .run(capture_stdout=True)
    )
    audio_array = np.frombuffer(audio, dtype=np.float32)
    # based on number of channels, re-arrange the array
    assert channels <= 2, "Cannot cope with more than two channels yet"
    if channels == 2:
        audio_array = (audio_array[0::2] + audio_array[1::2]) / 2        
    return audio_array, sr

def find_highest_elements(arr1, arr2):
    """
    returns an array of the same length as arr1 and arr2
    for each step in the new array, if arr1[step] > arr2[step], write 0, otherwise, 1
    """
    if len(arr1) != len(arr2):
        raise ValueError("Input arrays must have the same length")
    result = [0 if l1 > l2 else 1 for l1, l2 in zip(arr1, arr2)]
    return np.array(result)

def process_repeats(input, thresh = 100):
    """
    work through values in input, writing current value 
    to an output
    once you've seen any value > thresh times, 
    switch to that value as your output value 
    and reset counts so the race to take over starts again
    """
    vals = np.unique(input)
    counts = {}
    for v in vals: counts[v] = 0 
    current = vals[0]
    output = np.zeros(len(input))
    for i in range(len(input)):# iterate input
        counts[input[i]] += 1 # log count for input at step i
        for k in counts.keys(): # look for count > thresh
            if counts[k] > thresh: # value k is over thresh
                current = k
                # reset counters
                for k in counts.keys(): counts[k] = 0
        output[i] = current
    return output




def test_find_highest_elements():
    # Example usage:
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([5, 4, 3, 2, 1])
    
    result = find_highest_elements(arr1, arr2)
    print("Comparison Result:", result)



def count_repeats(arr):
    """
    used to create the segment descriptors: [[0, 123], [1, 2323]]
    where the first value is the array value, the second is the number of repeats
    iterates over the sent 1D array, counting the length of repeating sequences
    when the value changes, counter is set to zero
    outputs a list of values encountered and how many repeats of each it sees
    """
    lens = []
    count = 0
    val = arr[0]
    for x in arr:
        if x == val: # no change
            count += 1
        else: # change
            lens.append([val, count])
            val = x
            count = 0
    return lens
    
def create_video_edit_moviepy(video_paths, segment_descriptors):
    clips = []
    offset = 0  # Initialize the offset
    videos = [VideoFileClip(p) for p in video_paths]
    #for video_index, duration in segment_descriptors:
    for seg_pair in segment_descriptors:
        video_index = int(seg_pair[0])
        duration = seg_pair[1]
        # video_path = video_paths[video_index]
        # clip = VideoFileClip(video_path).subclip(offset, offset + duration)
        clip = videos[video_index].subclip(offset, offset + duration)
        clips.append(clip)
        offset += duration  # Accumulate the offset for the next segment

    final_clip = concatenate_videoclips(clips)
    # final_clip.write_videofile(output_path, codec='libx264')
    
    print(f'Video edit created')
    return final_clip


def remux_video(video_file, audio_file):
    input_video = video_file
    new_audio = audio_file
    output_video = video_file[0:-4] + "-mixed" + video_file[-4:]
    
    # Define the input video and audio streams
    input_video_stream = ffmpeg.input(input_video)
    input_audio_stream = ffmpeg.input(new_audio)
    
    # Use the 'map' option to copy the video stream and the audio stream
    output_stream = ffmpeg.output(input_video_stream['v'], input_audio_stream['a'], output_video, c='copy')
    
    # Run the ffmpeg command to replace the audio using the copy codec
    ffmpeg.run(output_stream, overwrite_output=True)



def buffer_to_audioclip(buffer, sr=48000):
    """
    converts the sent buffer of samples into a moviepy audio clip
    by writing to a file then reading back in
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio_file:
        temp_audio_filename = temp_audio_file.name
        # this write should retain the buffer data as 32 bit floats
        wavfile.write(temp_audio_filename, sr, buffer)
    # 
    audio_clip = AudioFileClip(temp_audio_filename) 
    # os.remove(temp_audio_filename)
    return audio_clip


def apply_gate(audio_data, sr=48000, gate_threshold_db=-25):
    board = Pedalboard([
        #Chorus(), Reverb(room_size=0.25)
        NoiseGate(gate_threshold_db, release_ms=200, attack_ms=10)
    ])
    effected = board(audio_data, sr, reset=True)
    return effected 

 # class pedalboard.Limiter(threshold_db: float = -10.0, release_ms: float = 100.0)
def apply_limit(audio_data, sr=48000, limit_threshold_db=-6):
    board = Pedalboard([
        #Chorus(), Reverb(room_size=0.25)
        Limiter(threshold_db=limit_threshold_db)#, release_ms=300, attack_ms=10)
    ])
    effected = board(audio_data, sr, reset=True)
    return effected 
    
def normalise(audio):
    # Find the maximum absolute value in the array
    max_abs_value = np.max(np.abs(audio))
    print("adding gain", 1/max_abs_value)
    # Normalize the array to the range -1 to 1
    normalized_array = audio / max_abs_value
    return normalized_array 

def loudness_stats(samples, block_size=512):
    offset = 1e-6
    samples = samples + offset
    
    # Calculate the number of blocks
    num_blocks = len(samples) // block_size
    
    # Initialize an array to store the RMS values
    rms_values = np.zeros(num_blocks)
    
    # Calculate RMS in blocks
    for i in range(num_blocks):
        block = samples[i * block_size : (i + 1) * block_size]
        rms_values[i] = np.sqrt(np.mean(block ** 2))

    # Step 1: Calculate the loudness (in dB) for each sample
    loudness_dB = 20 * np.log10(np.abs(rms_values))
    # Step 2: Compute the mean (average) of the loudness values
    mean_loudness = np.mean(loudness_dB)
    
    # Step 3: Calculate the standard deviation of the loudness values
    std_deviation_loudness = np.std(loudness_dB)
    print(f"Mean loudness in dB: {mean_loudness:.2f} dB")
    print(f"Standard deviation of loudness in dB: {std_deviation_loudness:.2f} dB")
    return {"rms_mean":mean_loudness, "rms_std":std_deviation_loudness}


def loudness_stats_old(samples):
    samples = np.array(samples)
    # print("analysing samples: ", len(samples))
    # stop zeros...
    offset = 1e-6
    samples = samples + offset
    # Step 1: Calculate the loudness (in dB) for each sample
    loudness_dB = 20 * np.log10(np.abs(samples))
    
    # Step 2: Compute the mean (average) of the loudness values
    mean_loudness = np.mean(loudness_dB)
    
    # Step 3: Calculate the standard deviation of the loudness values
    std_deviation_loudness = np.std(loudness_dB)
    
    print(f"Mean loudness in dB: {mean_loudness:.2f} dB")
    print(f"Standard deviation of loudness in dB: {std_deviation_loudness:.2f} dB")
    return {"rms_mean":mean_loudness, "rms_std":std_deviation_loudness}


# def loudness_stats(buffer):
#     # Step 1: Calculate the root mean square (RMS)
#     rms = np.sqrt(np.mean(buffer ** 2))
#     rmsd = np.sqrt(np.std(buffer ** 2))
    
#     # Step 2: Convert RMS to dB
#     loudness_dB = 20 * np.log10(rms)
#     loudness_dB_std = 20 * np.log10(rmsd)
    
#     print(f"Median loudness in dB: {loudness_dB:.2f} dB")
#     print(f"SD: {loudness_dB_std:.2f} dB")
#     return loudness_dB
   

import sys

assert len(sys.argv) == 6, "## args: input video 1, input video 2, output video, output_audio_processed, output_audio_clean"

print("Checking config")

### some params to read from the CLI probably
sr = 48000
window_size = round(0.5 * sr)  
hits_to_switch = sr # how many 'loudest frame' hits needed to switch to that track? 

video_in_file_1 = sys.argv[1] # 
video_in_file_2 = sys.argv[2]
video_out_file = sys.argv[3]
proc_audio_file = sys.argv[4]
clean_audio_file = sys.argv[5]

# work out the format 
video_ext = video_out_file[-3:]
video_codec = None
audio_codec = None

is_apple_os = platform.system() == 'Darwin'

if video_ext == "mov":
    video_codec = "prores" # no videotoolbox for prores yet, not sure why 
    audio_codec = "pcm_s24le"
elif video_ext == "mp4":
    if is_apple_os:
        video_codec = "h264_videotoolbox" # fast one for mac
    else:
        video_codec = "h264" # fast one for mac
    audio_codec = "aac"

assert video_codec is not None, "Video format not supported" + video_ext

print("V1:", video_in_file_1, "V2:", video_in_file_2, "V_out:", video_out_file, "v/a", video_codec, audio_codec, "Audio clean:",clean_audio_file, "Audio processed:", proc_audio_file)

#assert False, "done"


video_paths = [video_in_file_1, video_in_file_2]
assert os.path.exists(video_paths[0]), video_paths[0] + " not found "
assert os.path.exists(video_paths[1]), video_paths[1] + " not found "


## compute moving average
clip1,sr = extract_audio_data(video_paths[0])
clip2,sr = extract_audio_data(video_paths[1])
print("Computing moving average for", (len(clip1) / sr), "seconds")

# moving average
clip1_ma = scipy.signal.convolve(np.abs(clip1), np.ones(window_size)/window_size, mode='valid')
clip2_ma = scipy.signal.convolve(np.abs(clip2), np.ones(window_size)/window_size, mode='valid')

print("Selecting loudest clip for", (len(clip1) / sr), "seconds")
## find the highest-> [0, 0, 1, 1, ...] per audio sample
highest_inds = find_highest_elements(clip1_ma, clip2_ma)
## apply repeater filer
print("Filtering short repeats")
highest_inds = process_repeats(highest_inds, thresh=hits_to_switch)
## verify there are some switches happening 
assert len(np.unique(highest_inds)) > 1, "Problem with highest inds - only one highest ind: " + str(np.unique(highest_inds))
print("Counting repeat lengths for", (len(highest_inds) / sr), "seconds")
## count repeats
item_reps_all = count_repeats(highest_inds)
print("Reps add up to ", (np.sum([v[1] for v in item_reps_all]) / sr), "seconds")

print(item_reps_all)

####################### EDIT THE VIDEO 
#######################

print("Analysis complete. Carrying out edits ")

# convert to fractions of a second
item_reps_secs = [[v[0], v[1]/sr] for v in item_reps_all]
clip = create_video_edit_moviepy(video_paths, item_reps_secs)

####################### NORMALISE, LIMIT AND GATE THE SOUND 
#######################

# noramlise
print("normalise")
clip1_proc = normalise(clip1)
clip2_proc = normalise(clip2)
print("limiter")
clip1_proc = apply_limit(clip1)
clip2_proc = apply_limit(clip2)

print("Compute loudness stats for gate")
stats1 = loudness_stats(clip1_proc)
stats2 = loudness_stats(clip2_proc)

print("gate")
clip1_proc = apply_gate(clip1_proc, gate_threshold_db=stats1["rms_mean"])
clip2_proc = apply_gate(clip2_proc, gate_threshold_db=stats2["rms_mean"])


# mix channels
print("mixing")
mixed_audio = (np.array(clip1_proc) + np.array(clip2_proc)) / 2
mixed_audio = normalise(mixed_audio)

print("mixing and clean audio")
# make an unprocessed version of the audio (aside from normalising)
mixed_audio_clean = (np.array(normalise(clip1)) + np.array(normalise(clip2))) / 2
mixed_audio_clean = normalise(mixed_audio)

print("writing audio files")
audio_as_clip = buffer_to_audioclip(mixed_audio)
audio_as_clip.write_audiofile(proc_audio_file)
audio_as_clip_clean = buffer_to_audioclip(mixed_audio)
audio_as_clip_clean.write_audiofile(clean_audio_file)


print("writing video file")
clip.write_videofile(video_out_file, codec=video_codec, audio_codec=audio_codec)

print("Remuxing audio to video file")

remux_video(video_out_file, proc_audio_file)
