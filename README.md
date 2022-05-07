# PBL (preference-based learning)
This repository houses the code utilized to run the experiments in 'A Preference-Based Learning Framework for Optimizing and Characterizing User Comfort during Dynamically Stable Crutch-less Exoskeleton Walking'


## System requirements:
This code relies on MATLAB (it has been tested on R2019b but will likely work for other versions). Other than installing MATLAB, no other installation is required. Older versions of the code are available in python for preference optimization [here](https://github.com/ernovoseller/CoSpar) and preference characterization [here](https://github.com/kli58/ROIAL). 

## Demo
To begin the framework corresponding to the experiments with subjects with paraplegia, run the individual sections in the 'run_experiment.m' script. This script outlines how to 
1. Load the experiment settings
2. Instanteate the PBL framework class
3. Begin the experiment with preference characterization
4. Continue the experiment with preference optimization
5. Plot the learned Gaussian posterior
6. Update and plot a final posterior using the 'postProcess' function

Another script is included called 'setup_new' which outlines the key settings and parameter definitions required to begin a new preference-based learning experiment.

