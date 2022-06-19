import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#Notes on the format of the data folder
    #32 participant files (1 per), "Each participant file contains two arrays:
    #Array name,	Array shape,	    Array contents
    #data	    40 x 40 (will only be using first 32) x 8064	video/trial x channel x data  (Total: 12,902,400 points, will be using 10,321,920)
    #labels	    40 x 4	        video/trial x label (valence, arousal, dominance, liking)
    #"
#
def plotTrial(dataset,channelList, subjectNum, trialNum, outfile):
    fig, axs = plt.subplots(8,4)
    title = 'DEAP Subject '+subjectNum+' Trial '+str(trialNum)
    for i in range(32):
        axs.flatten()[i].plot(dataset['data'][trialNum][i])
        axs.flatten()[i].set_title(channelList[i])
    fig.suptitle(title)
    outfile.savefig(fig)
    plt.close('all')#added after discovering this when working on Welch code
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
    outputFileName = 'DEAP '+subjectNumber+'.pdf'
    outputFile = PdfPages(outputFileName)
    for trial in range(40):
        figTrial, axsTrial = plotTrial(dataSubject, channelList, subjectNumber, trial, outputFile)
    outputFile.close()
    print("Done")

if __name__ == "__main__":
    main()