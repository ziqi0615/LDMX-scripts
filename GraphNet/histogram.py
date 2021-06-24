import numpy as np
import glob
import uproot
import awkward
import matplotlib.pyplot as plt
import concurrent.futures
import copy
import matplotlib
import math
executor = concurrent.futures.ThreadPoolExecutor(12)

sig_base = '/home/pmasterson/GraphNet_input/v12/sig_extended_tracking/'
bkg_base = '/home/pmasterson/GraphNet_input/v12/bkg_12M/evaluation/'
bkg_files = glob.glob(bkg_base+'4gev_v12_pn_101_ldmx-det-v12_run1_seeds_2_3_None.root')
sig_1    = glob.glob(sig_base+'*0.001*.root')[:]
#sig_10   = glob.glob(sig_base+'*0.01*.root')[:]
#sig_100  = glob.glob(sig_base+'*0.1*.root')[:]
#sig_1000 = glob.glob(sig_base+'*1.0*.root')[:]
#sig_dict = {1:sig_1, 10:sig_10, 100:sig_100, 1000:sig_1000}


load_branches = ['EcalRecHits_v12.id_', 'EcalRecHits_v12.energy_', 'EcalScoringPlaneHits_v12.pdgID_', 'EcalScoringPlaneHits_v12.trackID_', 'EcalScoringPlaneHits_v12.x_', 'EcalScoringPlaneHits_v12.y_', 'EcalScoringPlaneHits_v12.z_', 'EcalScoringPlaneHits_v12.px_', 'EcalScoringPlaneHits_v12.py_', 'EcalScoringPlaneHits_v12.pz_']

# Functions
def projection(Recoilx, Recoily, Recoilz, RPx, RPy, RPz, HitZ):
    x_final = Recoilx + RPx/RPz*(HitZ - Recoilz)
    y_final = Recoily + RPy/RPz*(HitZ - Recoilz)
    return (x_final, y_final)

def dist(p1, p2):
    return math.sqrt(np.sum( ( np.array(p1) - np.array(p2) )**2 ))

# Constants
scoringPlaneZ = 240.5
ecalFaceZ = 248.35
cell_radius = 5.0


def get_fX_fY(filelist):
    # First, read cellMap for fiducial code:
    cellMap = {}
    for i, x, y in np.loadtxt('data/v12/cellmodule.txt'):
        cellMap[i] = (x, y)
   
    cells = list(cellMap.values())

    print("Loaded detector info")

    print("Reading files")

    total_events = 0
    total_nonfiducial = 0

    for f in filelist:

        print("    Reading file {}".format(f))
        t = uproot.open(f)['LDMX_Events']
        if len(t.keys()) == 0:
            print("    File empty, skipping")
        table_temp = t.arrays(expressions=load_branches, interpretation_executor=executor)
        table = {}
        for k in load_branches:
            table[k] = table_temp[k]
        
        print('Starting selection')

        total_events += len(table["EcalScoringPlaneHits_v12.x_"])

        for event in range(len(table["EcalScoringPlaneHits_v12.x_"])):       
            
            if (event % 5000 == 0):
                print('Loading Event ' + str(event)) 
            
            for hit in range(len(table["EcalScoringPlaneHits_v12.x_"][event])):
                 
                if ((table["EcalScoringPlaneHits_v12.pdgID_"][event][hit] == 11) and \
                   (table["EcalScoringPlaneHits_v12.z_"][event][hit] > 240) and \
                   (table["EcalScoringPlaneHits_v12.z_"][event][hit] < 241.001) and \
                   (table["EcalScoringPlaneHits_v12.pz_"][event][hit] > 0) and \
                   (table["EcalScoringPlaneHits_v12.trackID_"][event][hit] == 1)):
                    
                    recoilX = table["EcalScoringPlaneHits_v12.x_"][event][hit]      
                    recoilY = table["EcalScoringPlaneHits_v12.y_"][event][hit]
                    recoilPx = table["EcalScoringPlaneHits_v12.px_"][event][hit]
                    recoilPy = table["EcalScoringPlaneHits_v12.py_"][event][hit]
                    recoilPz = table["EcalScoringPlaneHits_v12.pz_"][event][hit]
                          
                    finalXY = projection(recoilX, recoilY, scoringPlaneZ, recoilPx, recoilPy, recoilPz, ecalFaceZ)                     
                     
                    fiducial = False
                    if not recoilX == -9999 and not recoilY == -9999 and not recoilPx == -9999 and not recoilPy == -9999:
                       
                        for cell in range(len(cells)):
                            celldis = dist(cells[cell], finalXY) 
                            if celldis <= cell_radius:
                                fiducial = True
                                break 

                    if fiducial == False:
                        total_nonfiducial += 1

        print("Total number of events: " + str(total_events))
        print("Total number of non-fiducial events: " + str(total_nonfiducial))

get_fX_fY(sig_1)
#get_fX_fY(sig_10)
#get_fX_fY(sig_100)
#get_fX_fY(sig_1000)
#get_fX_fY(bkg_files)
#print("Done.  Plotting ECAL Face Hits...")
#my_cmap = plt.cm.jet
#my_cmap.set_under('white', 1)
#plt.figure()
#plt.hist2d(fX_bkg, fY_bkg, bins=500, range=([-300,300],[-300,300]), cmin = 1,  cmap=my_cmap, norm=matplotlib.colors.LogNorm())
#plt.colorbar()
#plt.xlabel('X (mm)')
#plt.ylabel('Y (mm)')
#plt.savefig('/home/dgj1118/LDMX-scripts/GraphNet/1MeV.png')
