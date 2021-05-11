import numpy as np
import glob
import uproot
import awkward
import matplotlib.pyplot as plt
import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(12)

bkg = '/home/aechavez/flattrees_for_david/bkg_tree.root'
bkg_files = glob.glob(bkg)

load_branches = ['recoilX','recoilY','recoilPx','recoilPy','recoilPz']

scoringPlaneZ = 240.5015
ecalFaceZ = 248.35
cell_radius = 5

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

def getXY(filelist):
    
    print('STARTING PROCESS')
    
    fX = []
    fY = []

    for f in filelist:
         print('    Reading /home/aechavez/flattrees_for_david/bkg_tree.root')
         t = uproot.open(f)['EcalVeto']
         if len(t.keys()) == 0:
            print("    File empty, skipping")
        
         table_temp = t.arrays(expressions=load_branches, interpretation_executor=executor)
         table = {}
         for k in load_branches:
             table[k] = table_temp[k]
            
         for i in range(len(table['recoilX'])):
        
              recoilX  = table['recoilX'][i]
              recoilY  = table['recoilY'][i]
              recoilPx = table['recoilPx'][i]
              recoilPy = table['recoilPy'][i]
              recoilPz = table['recoilPz'][i]
          
              recoilfX = CallX(ecalFaceZ, recoilX, recoilY, scoringPlaneZ, recoilPx, recoilPy, recoilPz)
              recoilfY = CallY(ecalFaceZ, recoilX, recoilY, scoringPlaneZ, recoilPx, recoilPy, recoilPz)
         
              fX.append(recoilfX)
              fY.append(recoilfY)
         
    return fX, fY

X, Y =  getXY(bkg_files)

print('Done. Plotting...')

plt.figure()

plt.hist2d(X, Y, bins=200, range=([-300,300], [-300,300]), cmap = 'jet', cmin = 1, vmin = 1)
plt.colorbar()
plt.xlabel('X (mm)')
plt.ylabel('Y (mm)')
plt.savefig('/home/dgj1118/LDMX-scripts/GraphNet/XYHits_v9.png')

