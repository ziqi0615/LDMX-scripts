import numpy as np
import glob
import uproot
import awkward
import matplotlib.pyplot as plt
import concurrent.futures
import copy
executor = concurrent.futures.ThreadPoolExecutor(12)

sig_base = '/home/pmasterson/GraphNet_input/v12/sig_extended_tracking/'
bkg_base = '/home/pmasterson/GraphNet_input/v12/bkg_12M/'
bkg_files = glob.glob(bkg_base+'*.root')[:1]
sig_1    = glob.glob(sig_base+'*0.001*.root')[:2]
sig_10   = glob.glob(sig_base+'*0.01*.root')[:2]
sig_100  = glob.glob(sig_base+'*0.1*.root')[:2]
sig_1000 = glob.glob(sig_base+'*1.0*.root')[:2]
sig_dict = {1:sig_1, 10:sig_10, 100:sig_100, 1000:sig_1000}


load_branches = ['EcalRecHits_v12.id_', 'EcalRecHits_v12.energy_', 'EcalScoringPlaneHits_v12.px_', 'EcalScoringPlaneHits_v12.py_',
                 'EcalScoringPlaneHits_v12.pdgID_', 'EcalScoringPlaneHits_v12.x_', 'EcalScoringPlaneHits_v12.y_', 'EcalScoringPlaneHits_v12.z_',
                 'EcalScoringPlaneHits_v12.px_', 'EcalScoringPlaneHits_v12.py_', 'EcalScoringPlaneHits_v12.pz_']


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
cell_radius = 5


def get_fX_fY(filelist):
    # First, read cellMap for fiducial code:
    cellMap = {}
    for i, x, y in np.loadtxt('data/v12/cellmodule.txt'):
        cellMap[i] = (x, y)
    print("Loaded detector info")

    print("Reading files")
    
    fX = [] # x-values
    fY = [] # y-values
    fX_length = 0
    fY_length = 0

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
        
        for i in range(len(table["EcalVeto_v12.nReadoutHits_"])):
            
            for j in range(len(table["EcalScoringPlaneHits_v12.px_"][i])):
                maxPz = 0
                if (table['EcalScoringPlaneHits_v12.pdgID_'][i][j] == 11) and \
                   (table['EcalScoringPlaneHits_v12.z_'][i][j] > 240) and \
                   (table['EcalScoringPlaneHits_v12.z_'][i][j] < 241) and \
                   (table['EcalScoringPlaneHits_v12.pz_'][i][j] > maxPz):
                    
                    maxPz = table['EcalScoringPlaneHits_v12.pz_'][i][j]

                    recoilX  = table['EcalScoringPlaneHits_v12.x_'][i][j]
                    recoilY  = table['EcalScoringPlaneHits_v12.y_'][i][j]
                    recoilPx = table['EcalScoringPlaneHits_v12.px_'][i][j]
                    recoilPy = table['EcalScoringPlaneHits_v12.py_'][i][j]
                    recoilPz = table['EcalScoringPlaneHits_v12.pz_'][i][j]

                    recoilfX = CallX(ecalFaceZ, recoilX, recoilY, scoringPlaneZ, recoilPx, recoilPy, recoilPz)
                    recoilfY = CallY(ecalFaceZ, recoilX, recoilY, scoringPlaneZ, recoilPx, recoilPy, recoilPz)
                    
            fX.append(recoilfX)
            fX_length += 1
            fY.append(recoilfY)
            fY_length += 1

    return fX, fY, fX_length, fY_length

# photonuclear background x and y hits
fX_bkg, fY_bkg, fX_bkg_length, fY_bkg_length = get_fX_fY(bkg_files)

# signal x and y hits
#fX_sig1, fY_sig1 = get_fX_fY(sig_1)
#fX_sig10, fY_sig10 = get_fX_fY(sig_10)
#fX_sig100, fY_sig100 = get_fX_fY(sig_100)
#fX_sig1000, fY_sig1000 = get_fX_fY(sig_1000)
print('The length of fX is ' + str(fX_bkg_length))
print('The length of fY is ' + str(fY_bkg_length))

print("Done.  Plotting...")
'''
for i in range(len(fY_bkg)):
    fY_bkg[i] = np.ma.masked_where(fY_bkg[i] == 0, fY_bkg[i])
for j in fX_bkg:
    fX_bkg[j] = np.ma.masked_where(fX_bkg[j] == 0, fX_bkg[j])
'''
#cmap = copy.copy(plt.cm.get_cmap("jet"))
#cmap = cmap.set_bad(color='white')
my_cmap = plt.cm.jet
my_cmap.set_under('white', 1)

# plotting the 2d histograms for background and signal
plt.figure()

plt.hist2d(fX_bkg, fY_bkg, bins=100, range=([-300,300],[-300,300]), cmin = 1,  cmap=my_cmap, vmin = 1)


#plt.hist2d(fX_sig1, fY_sig1, label='1 MeV')

#plt.hist2d(fX_sig10, fY_sig10, label='10 MeV')

#plt.hist2d(fX_sig100, fY_sig100, label='100 MeV')

#plt.hist2d(fX_sig1000, fY_sig1000, label='1000 MeV')
plt.colorbar()
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.savefig('/home/dgj1118/LDMX-scripts/GraphNet/XYHits.png')
#plt.show()

