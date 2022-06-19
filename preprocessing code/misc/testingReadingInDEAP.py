import pickle
import matplotlib.pyplot as plt
#Notes on the format of the data folder
    #32 participant files (1 per), "Each participant file contains two arrays:
    #Array name,	Array shape,	    Array contents
    #data	    40 x 40 (will only be using first 32) x 8064	video/trial x channel x data
    #labels	    40 x 4	        video/trial x label (valence, arousal, dominance, liking)
    #"
#
def main():
    #testing with participant 1, reading in participant 1's data
    with open('preprocessing code\DEAP_data_preprocessed_python\s01.dat','rb') as f:
        dataSubject1 = pickle.load(f, encoding='latin1')
    #list of the names of all the channels
    channelList = ['Fp1','AF3','F3','F7',
    'FC5','FC1','C3','T7',
    'CP5','CP1','P3','P7',
    'PO3','O1','Oz','Pz',
    'Fp2','AF4','Fz','F4',
    'F8','FC6','FC2','Cz',
    'C4','T8','CP6','CP2',
    'P4','P8','PO4','O2'] 
    fig, axs = plt.subplots(8,4)
    #starting off by plotting 0,0
    for i in range(32):
        axs.flatten()[i].plot(dataSubject1['data'][0][i])
        axs.flatten()[i].set_title(channelList[i])
    fig.suptitle('DEAP Subject 1 Trial 0')
    plt.show()
    print("Done")

if __name__ == "__main__":
    main()