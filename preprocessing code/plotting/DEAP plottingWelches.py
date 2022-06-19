import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.signal as signal

#Notes on the format of the data folder
    #32 participant files (1 per), "Each participant file contains two arrays:
    #Array name,	Array shape,	    Array contents
    #data	    40 x 40 (will only be using first 32) x 8064	video/trial x channel x data (Total: 12,902,400 points, will be using 10,321,920)
    #labels	    40 x 4	        video/trial x label (valence, arousal, dominance, liking)
    #"
#

def plotTrialWelch(dataset,channelList, subjectNum, trialNum, outfile):
    fig, axs = plt.subplots(8,4)
    title = 'DEAP Subject '+subjectNum+' Trial '+str(trialNum)+ ' Welches '
    for i in range(32):
        frequencies, powers = signal.welch(dataset['data'][trialNum][i], fs=128, nperseg=1024)
        axs.flatten()[i].plot(frequencies,powers)
        axs.flatten()[i].set_title('Welch for '+channelList[i])
    fig.suptitle(title)
    outfile.savefig(fig)
    plt.close('all')
    return fig,axs

def main():
    subjectNumber = input("Enter the number of the subject to plot (1-32): ")
    #testing with participant 1, reading in participant 1's data
    if int(subjectNumber) < 10:
        fileName = 'preprocessing code\DEAP_data_preprocessed_python\s0'+subjectNumber+'.dat'
    else:
        fileName = 'preprocessing code\DEAP_data_preprocessed_python\s'+subjectNumber+'.dat'
    with open(fileName,'rb') as f:
        dataSubject = pickle.load(f, encoding='latin1')
    #list of the names of all the channels
    channelList = ['Fp1','AF3','F3','F7',
    'FC5','FC1','C3','T7',
    'CP5','CP1','P3','P7',
    'PO3','O1','Oz','Pz',
    'Fp2','AF4','Fz','F4',
    'F8','FC6','FC2','Cz',
    'C4','T8','CP6','CP2',
    'P4','P8','PO4','O2'] 
    outputFileName = 'DEAP '+subjectNumber+' welch.pdf'
    outputFile = PdfPages(outputFileName)
    for trial in range(40):
        figTrial, axsTrial = plotTrialWelch(dataSubject, channelList, subjectNumber, trial, outputFile)
    outputFile.close()
    print("Done")

if __name__ == "__main__":
    main()