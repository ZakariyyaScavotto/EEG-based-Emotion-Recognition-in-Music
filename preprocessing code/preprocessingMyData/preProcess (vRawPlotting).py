from scipy.fft import fft, ifft
import scipy.signal
import pandas as pd
import matplotlib.pyplot as plt
# #print(pd.__version__) #works, 1.3.3
data = pd.read_excel(r'3_29 30s Pre3.xlsx', usecols='D:Q',header=1) #imports the excel file (using this one to test)

fig, axs = plt.subplots(5,3) #Graphs raw data for each sensor in a subplot 
for i in range(14):
    axs[i%5,i%3].plot(data.iloc[:,i])
    axs[i%5,i%3].set_title(data.columns[i])
plt.show()

