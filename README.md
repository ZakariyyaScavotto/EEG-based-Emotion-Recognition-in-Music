# EEG-based-Emotion-Recognition-in-Music

Finalized repository of all my code for my high school senior research project, EEG-based Emotion Recognition in Music using Machine Learning. For ease of understanding, below is an explanation of the repository's structure:

- DataReadingTesting: contains notes and code from me testing streaming data from my EMOTIV EPOC+ EEG headset to a Python program using EMOTIV's Lab Streaming Layer (LSL).
- DEAPBazgir: : This contains the preprocessed version of DEAP processed using the Discrete Wavelet Transform (DWT) extraction method. Each file represents the results for a different subject.
- DEAPBazgirPCA: This contains the preprocessed version of DEAP processed using the DWT extraction method, with a PCA conducted on it. Each file represents the results for a different subject.
- DEAPDiffEntropyALLBANDS: This contains the preprocessed DEAP with the Differential Entropy (DE) extracted across all EEG subbands. Each file represents the results for a different subject.
- DEAPDiffEntropySUBBANDS: This contains the preprocessed DEAP with the DE extracted for each EEG subband. Each file represents the results for a different subject.
- DEAPwelched: This contains the preprocessed DEAP using Welch’s method for feature extraction. Each file represents the results for a different subject.
- eegEmotionsandMusicFlask(Final Product): This folder contains the code for my final demonstration application, which can be run by executing “python app.py” while in this folder from the command line.
- FinalSVMs: This folder contains the finalized trained SVMs for my project saved utilizing the joblib library.
- MLCode: This was the primary folder that I saved my ML code in.
  - DBN: This folder contains the code from when I tried training Deep Belief Networks (DBN) (implemented from this Github repo: https://github.com/albertbup/deep-belief-network) on the EEG data.
  - DeepSITH and SIF_Capstone: Contains code from when I tried training DeepSITH (code from https://github.com/gauvand/SIF_Capstone, https://github.com/compmem/SITH_Layer, https://github.com/compmem/DeepSITH) on the EEG data.
  - SVM: This contains the code I used when working on training SVMs on the EEG data processed using the different feature extraction methods I used.
- preprocessing code: This was the primary folder I saved my data processing code in
  - NOTE: In order to work with the preprocessing code, you will need to request access to the DEAP dataset and download it for use (https://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html).
- RawDEAPBazgir: This contains the results of the raw version of the DEAP data being processed using the DWT extraction method, and saved using the pickle library. Each file represents the results for a different subject.
- .gitignore and README.md
