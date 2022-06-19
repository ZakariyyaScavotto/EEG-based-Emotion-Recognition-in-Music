import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
data = pd.read_excel(r'preprocessing code/3_29 30s Pre3.xlsx', usecols='D:Q',header=1) #imports the excel file (using this one to test)

fig, axs = plt.subplots(5,3) #Graphs raw data for each sensor in a subplot 
for i in range(14):
    axs.flatten()[i].plot(data.iloc[:,i])
    axs.flatten()[i].set_title(data.columns[i])
fig.suptitle('Raw Data')
#2-40Hz bandpass filter
fig2, axs2 = plt.subplots(5,3) #Graphs raw data for each sensor in a subplot 
lowPassCutoff, highPassCutoff = 2/(0.5*128), 40/(0.5*128) #cutoff = cutoffFrequency(Hz)/(0.5*sample rate (emotiv records at sample rate of 128Hz))
b, a = signal.butter(1, [lowPassCutoff, highPassCutoff], btype='bandpass',output='ba')
for i in range(14):
    data.iloc[:,i] = signal.filtfilt(b,a, data.iloc[:,i])
    axs2.flatten()[i].plot(data.iloc[:,i])
    axs2.flatten()[i].set_title(data.columns[i])
fig2.suptitle('2-40Hz Band-Pass Filtered Data')
#welch 
fig3, axs3 = plt.subplots(5,3)
for i in range(14):
    frequencies, powers = signal.welch(data.iloc[:,i], fs=128, nperseg=1024)
    axs3.flatten()[i].plot(frequencies, powers)
    axs3.flatten()[i].set_title('Welch For ' + data.columns[i])
plt.show()
print("Done")