"""Example program to show how to read a multi-channel time series from LSL."""
#Modified from the example on the Emotiv LSL Github
from pylsl import StreamInlet, resolve_stream

# first resolve an EEG stream on the lab network
print("looking for a stream...")
streams = resolve_stream('type', 'EEG')
print(streams)

# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

while True:
    # get a new sample (you can also omit the timestamp part if you're not
    # interested in it)
    sample = inlet.pull_sample()
    #sample = inlet.pull_chunk()
    #print(sample[0][1])

#1 thing in chunk
#[1638555458.1816, 127.0, 0.0, 4582.949, 4142.051, 4169.103, 4182.949, 4172.821, 4163.462, 4131.538, 4271.154, 4168.333, 4163.846, 5232.179, 4178.59, 4157.821, 4174.359,0,0]
#timestamp, count, ?, 14 channels (what order?), ?, ?
#According to https://emotiv.gitbook.io/emotivpro-v2-0/lab-streaming-layer-lsl/lsl-outlet-configuration ["Timestamp", "Counter", "Interpolate", <EEG sensors>, "HardwareMarker" ]

#for epoc+ https://emotiv.gitbook.io/cortex-api/data-subscription/data-sample-object#eeg
# [
#     "COUNTER",
#     "INTERPOLATED",
#     "AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4",
#     "RAW_CQ",
#     "MARKER_HARDWARE",
#     "MARKERS"
# ]

#sample
#[1638556193.0565, 84.0, 0.0, 4061.667, 4175.897, 4189.744, 4200.641, 4186.667, 4198.59, 4195.256, 4186.795, 4201.667, 4180.0, 4219.615, 4181.41, 4161.282, 4049.359,0.0,0.0]
#the channels are indicies 3-16, but what order?