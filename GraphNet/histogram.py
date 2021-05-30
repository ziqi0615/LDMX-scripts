import numpy as np
import glob
import uproot
import awkward
import matplotlib.pyplot as plt
import concurrent.futures
import copy
import matplotlib
executor = concurrent.futures.ThreadPoolExecutor(12)

sig_base = '/home/pmasterson/GraphNet_input/v12/sig_extended_tracking/'
bkg_base = '/home/pmasterson/GraphNet_input/v12/bkg_12M/evaluation/'
bkg_files = glob.glob(bkg_base+'4gev_v12_pn_101_ldmx-det-v12_run1_seeds_2_3_None.root')
#sig_1    = glob.glob(sig_base+'*0.001*.root')[:2]
#sig_10   = glob.glob(sig_base+'*0.01*.root')[:2]
#sig_100  = glob.glob(sig_base+'*0.1*.root')[:2]
#sig_1000 = glob.glob(sig_base+'*1.0*.root')[:2]
#sig_dict = {1:sig_1, 10:sig_10, 100:sig_100, 1000:sig_1000}


load_branches = ['EcalRecHits_v12.id_', 'EcalRecHits_v12.energy_', 'EcalScoringPlaneHits_v12.pdgID_', 'EcalScoringPlaneHits_v12.trackID_', 'EcalScoringPlaneHits_v12.x_', 'EcalScoringPlaneHits_v12.y_', 'EcalScoringPlaneHits_v12.z_', 'EcalScoringPlaneHits_v12.px_', 'EcalScoringPlaneHits_v12.py_', 'EcalScoringPlaneHits_v12.pz_']


veto_branches = ['nReadoutHits_', 'electronContainmentEnergy_', 'photonContainmentEnergy_', 'summedDet_', 'summedTightIso_']


def CallX(Hitz, Recoilx, Recoily, Recoilz, RPx, RPy, RPz):
    Point_xz = [Recoilx, Recoilz]
    #Almost never happens
    if RPx == 0:
        slope_xz = 99999
    else:
        slope_xz = RPz / RPx

    x_val = (float(Hitz - Point_xz[1]) / float(slope_xz)) + Point_xz[0]
    return x_val

def CallY(Hitz, Recoilx, Recoily, Recoilz, RPx, RPy, RPz):
    Point_yz = [Recoily, Recoilz]
    #Almost never happens
    if RPy == 0:
        slope_yz = 99999
    else:
        slope_yz = RPz / RPy

    y_val = (float(Hitz - Point_yz[1]) / float(slope_yz)) + Point_yz[0]
    return y_val

scoringPlaneZ = 240.5015
ecalFaceZ = 248.35
cell_radius = 5.0


def get_fX_fY(filelist):
    # First, read cellMap for fiducial code:
    cellMap = {}
    for i, x, y in np.loadtxt('data/v12/cellmodule.txt'):
        cellMap[i] = (x, y)
    print("Loaded detector info")

    print("Reading files")
    
    fX = [] # x-values Ecal Face
    fY = [] # y-values Ecal Face

    for f in filelist:
        print("    Reading file {}".format(f))
        t = uproot.open(f)['LDMX_Events']
        if len(t.keys()) == 0:
            print("    File empty, skipping")
        table_temp = t.arrays(expressions=load_branches, interpretation_executor=executor)
        table = {}
        for k in load_branches:
            table[k] = table_temp[k]
        # Load veto branches:
        EcalVeto = t["EcalVeto_v12"]
        for k in veto_branches:
            table["EcalVeto_v12."+k] = EcalVeto[k].array(interpretation_executor=executor)
        nHitsArr = awkward.sum(table[load_branches[1]] > 0, axis=1)
        
        print('Starting selection')

        #total_events = len(table["EcalVeto_v12.nReadoutHits_"])
        total_events = len(table["EcalScoringPlaneHits_v12.x_"])

        for i in range(len(table["EcalVeto_v12.nReadoutHits_"])):
                
            if (i % 100 == 0):
                print('Finished Event ' + str(i)) 
            
            if (i > 10000):
                break
            
            for j in range(len(table["EcalScoringPlaneHits_v12.px_"][i])):
            
                maxP = 0
                for k in range(len(table["EcalScoringPlaneHits_v12.px_"][i])):
                    squared = (table['EcalScoringPlaneHits_v12.pz_'][i][j])**2 + \
                    (table['EcalScoringPlaneHits_v12.px_'][i][j])**2 + \
                    (table['EcalScoringPlaneHits_v12.py_'][i][j])**2 
                    
                    if (table['EcalScoringPlaneHits_v12.pdgID_'][i][j] == 11) and \
                    (table['EcalScoringPlaneHits_v12.z_'][i][j] > 240) and \
                    (table['EcalScoringPlaneHits_v12.z_'][i][j] < 241) and \
                    (table['EcalScoringPlaneHits_v12.trackID_'][i][j] == 1) and \
                    (squared > (maxP)**2):
                     
                        maxP = np.sqrt(squared)

                    recoilX  = table['EcalScoringPlaneHits_v12.x_'][i][j]
                    recoilY  = table['EcalScoringPlaneHits_v12.y_'][i][j]
                    recoilPx = table['EcalScoringPlaneHits_v12.px_'][i][j]
                    recoilPy = table['EcalScoringPlaneHits_v12.py_'][i][j]
                    recoilPz = table['EcalScoringPlaneHits_v12.pz_'][i][j]
                   
                    recoilfX = CallX(ecalFaceZ, recoilX, recoilY, scoringPlaneZ, recoilPx, recoilPy, recoilPz)
                    recoilfY = CallY(ecalFaceZ, recoilX, recoilY, scoringPlaneZ, recoilPx, recoilPy, recoilPz) 
             
#            fX.append(recoilfX)
#            fY.append(recoilfY)
            
            fiducial = False
            if not recoilX == -9999 and not recoilY == -9999 and not recoilPx == -9999 and not recoilPy == -9999 and not recoilPz == -9999:
                for c_val in cellMap.values():
                    xdis = recoilfY - c_val[1]
                    ydis = recoilfX - c_val[0]
                    celldis = np.sqrt(xdis**2 + ydis**2)
                    if celldis <= cell_radius:
                        fiducial = True 

            if fiducial == False:
                fX.append(recoilfX)
                fY.append(recoilfY)
             
    return fX, fY, total_events

# photonuclear background x and y hits
fX_bkg, fY_bkg, events = get_fX_fY(bkg_files)
print("Total number of events: " + str(events))
print("Total number of selected events: " + str(len(fX_bkg)))
print("Done.  Plotting ECAL Face Hits...")
my_cmap = plt.cm.jet
my_cmap.set_under('white', 1)
plt.figure()
plt.hist2d(fX_bkg, fY_bkg, bins=500, range=([-300,300],[-300,300]), cmin = 1,  cmap=my_cmap, norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.savefig('/home/dgj1118/LDMX-scripts/GraphNet/test2.png')
