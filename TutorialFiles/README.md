# Quick Introduction to LDMX Software
Before you dive into the more in depth tutorials located in the sidebar on the right hand side of the [wiki](https://github.com/IncandelaLab/LDMX-scripts/wiki), here is a very quick guide on how to get some LDMX software working out-of-box!

## Installing pre-requisite software
* Install [XQuartz](https://www.xquartz.org/) (mac) or [Xming](http://www.straightrunning.com/XmingNotes/) (windows) 
* Install [Docker](https://docs.docker.com/engine/install/) and make an account 
* Ensure that you have a terminal app if you are on Mac or Linux, or Ubuntu (and WSL 2) downloaded if you are on Windows, then open up a terminal (shell/bash session)

## Installing LDMX Software Locally (i.e., on your own computer, no ssh or POD needed)
* Run the following commands (one line at a time) to install LDMX software
```
cd ~
git clone --recursive https://github.com/LDMX-Software/ldmx-sw.git 
source ldmx-sw/scripts/ldmx-env.sh
cd ldmx-sw; mkdir build; cd build;
ldmx cmake ..
ldmx make install
```
NOTE: You will need to run the source command every time you start a new terminal and want to use ldmx-sw, or you can add it to your ```.bashrc``` file (runs on startup) using ```vim```. You will also need your Docker desktop app running. 
* Now download LDMX-scripts
```
cd ~
# replace this next command with this repo for now (until I do a pull request)
# i.e., git clone -b v14_tutorial_files --recursive https://github.com/DuncanWilmot/LDMX-scripts LDMX-scripts-temp
git clone --recursive https://github.com/IncandelaLab/LDMX-scripts
```

## Generating Photonuclear (PN) Background samples
* We will use our ```v3_v14_pn_config.py``` file in the folder marked ```TutorialFiles``` in ```LDMX-scripts``` to generate some simulated samples using ldmx-sw v3 with v14 detector geometry. This will take a few minutes.
```
cd ~/LDMX-scripts/TutorialFiles
ldmx fire v3_v14_pn_config.py  
```
* The new samples will be saved as ```100k_v14_pn_testprod.root``` in ```~/LDMX-scripts/TutorialFiles```. Load ```ROOT``` and open a ```TBrowser``` to view them. You will need to first run the XQuartz or Xming app for this to work
```
ldmx root
# wait until you see root[0] in your terminal
new TBrowser()
```
This will open a graphical browser much like a file explorer but with plotting capabilities. Double click your root file to expand it and begin navigating through the branches/directories by double clicking them. When you get to LDMX variables with a leaf icon next to them, double clicking these will plot a histogram with their values

![image](https://github.com/DuncanWilmot/LDMX-scripts/assets/71404398/5ec4788f-31c7-43d5-ac0f-00cc32ec89ae)

![image](https://github.com/DuncanWilmot/LDMX-scripts/assets/71404398/9057a31e-2682-4c3a-85c7-543dfc588212)

## Flattening Trees for BDT
* You may need to download a couple extra ldmx-sw files before the next step. Go to the following directory and look for the files ```CaloTrigPrim.h``` and ```CaloCluster.h``` with the ```ls``` command. If you don't see them, execute the two ```wget``` commands
```
cd ~/ldmx-sw/Recon/include/Recon/Event
ls | grep -i '^calo[ct].*.h'
# if you don't see the two files execute the following two wget comands to download them
wget https://raw.githubusercontent.com/LDMX-Software/ldmx-sw/trunk/Recon/include/Recon/Event/CaloTrigPrim.h
wget https://raw.githubusercontent.com/LDMX-Software/ldmx-sw/trunk/Recon/include/Recon/Event/CaloCluster.h
```
* Now we will flatten the tree of the PN samples we just generated. This essentially just means picking all the leaves we want in the root file (e.g., nReadoutHits) and placing them in a single branch. This would be a bit like placing all the files you want to keep in your home directory, and deleting all subdirectories. In this case, we want all the ECal Veto variables needed for the BDT. We will accomplish this by using ```gabrielle_treeMaker.py``` on our samples ```100k_v14_pn_testprod.root```. The output with the flattened tree will be saved as ```100k_pn_v14_flatout.root```
```
cd ~/LDMX-scripts/TutorialFiles
# the following three lines are one single command!
ldmx python3 gabrielle_treeMaker.py \
-i $PWD/100k_v14_pn_testprod.root \
-o $PWD -g 100k_pn_v14_flatout
```

![image](https://github.com/DuncanWilmot/LDMX-scripts/assets/71404398/e0869bf0-bf22-47b9-bb7e-0c51357f0c14)

## Evaluate Gabrielle BDT
* We will now evaluate the pretrained v3 Gabrielle BDT on ```100k_pn_v14_flatout.root``` using a pickled set of weights in ```gabrielle_train_out_v3_weights.pkl``` obtained during training. The output will be saved as ```100k_pn_evalout.root```...Note: again, the three lines are one command, not three
```
ldmx python3 gabrielle_bdtEval.py \
-i $PWD/100k_pn_v14_flatout.root \
-o $PWD -g 100k_pn_v14_evalout
```
* Inspect the output with a ```TBrowser``` as before. You should see a new leaf called ```discValue_ECalVeto```. This discriminator value is a number between 0 and 1, which can more or less be interpreted as a measure of probability that an event is signal (so this number should be quite small for typical PN background events if your model is working well). We can set a threshold for this value and cut events accordingly to get rid of background while (ideally) preserving as much signal as possible.

## Plotting BDT Variables
DETAILED INSTRUCTIONS IN DEVELOPMENT...
For the time being, check out the [plotting scripts and README](https://github.com/IncandelaLab/LDMX-scripts/tree/master/plotting)





