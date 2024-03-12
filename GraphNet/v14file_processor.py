#!/home/pmasterson/miniconda3/envs/torchroot/bin/python

#SBATCH -n 20 -p short
#SBATCH --output=slurm_file_processor_2.out

# NOTE:  was --nodes=1, --ntasks-per-node 2
# NOTE:  Removed -p short

import numpy as np
import uproot
import awkward
import glob
import os
import re
print("Importing ROOT")
import ROOT as r
print("Imported root.  Starting...")
from multiprocessing import Pool

"""
file_processor.py

Purpose:  Read through ROOT files containing LDMX events that ParrticleNet will be trained on, drop every event that
doesn't pass the ParticleNet preselection, and save the remainder to output ROOT files that ParticleNet can read.
This was introduced because both the preselection and the pT calculation involve loading information from the ROOT
files that ParticleNet itself doesn't need, and would substantially increase runtime if ParticleNet performed the
calculation for every event individually.

Outline:
- For every input file, do the following:
   - Read all necessary data from the file into arrays using uproot.
   - Drop all events that fail the preselection condition.
   - Compute the pT of each event from the TargetScoringPlaneHit information (needed for pT bias plots, and not
     present in ROOT files), and keep track of it alongside the other arrays/branches loaded for the file.
   - Use ROOT to create new output files and to fill them with the contents of the loaded arrays.

"""

# Base:

# Directory to write output files to:
output_dir = 'processor_output'
# Used v12/signal_230_trunk, background_230_trunk for 2.3.0
# 3.0.0:
"""
file_templates = {
    0.001: '/home/pmasterson/events/v3.0.0_trigger/signal/*0.001*.root',  # 0.001 GeV, etc.
    0.01:  '/home/pmasterson/events/v3.0.0_trigger/signal/*0.01*.root',
    0.1:   '/home/pmasterson/events/v3.0.0_trigger/signal/*0.1*.root',
    1.0:   '/home/pmasterson/events/v3.0.0_trigger/signal/*1.0*.root',
    # Note:  m=0 here refers to PN background events
    0:     '/home/pmasterson/events/v3.0.0_trigger/background/*.root'
}
"""

# v13 geometry:
file_templates = {
    0.001:  '/home/vamitamas/Samples8GeV/Ap0.001GeV_sim/*.root',
    0.01:  '/home/vamitamas/Samples8GeV/Ap0.01GeV_sim/*.root',
    0.1:   '/home/vamitamas/Samples8GeV/Ap0.1GeV_sim/*.root',
    1.0:   '/home/vamitamas/Samples8GeV/Ap1GeV_sim/*.root',
    0:     '/home/vamitamas/Samples8GeV/v3.3.3_ecalPN*/*.root'    
}

# additional eval sample:  Used v12/sig_extended_extra, v12/bkg_12M/evaluation for 2.3.0



# Standard preselection values (-> 95% sig/5% bkg)
MAX_NUM_ECAL_HITS = 50 #60  #110  #Now MUCH lower!  >99% of 1 MeV sig should pass this. (and >10% of bkg)
MAX_ISO_ENERGY = 500  # NOTE:  650 passes 99.99% sig, ~13% bkg for 3.0.0!  Lowering...
# Results:  >0.994 vs 0.055

# Branches to save:
# Quantities labeled with 'scalars' have a single value per event.  Quantities labeled with 'vectors' have
# one value for every hit (e.g. number of ecal hits vs x position of each hit).
# (Everything else can be safely ignored)
# data[branch_name][scalars/vectors][leaf_name]
data_to_save = {
    'EcalScoringPlaneHits': {
        'scalars':[],
        'vectors':['pdgID_', 'layerID_','trackID_' ,'x_', 'y_', 'z_',
                   'px_', 'py_', 'pz_', 'energy_']
    },
    # NEW:  Added to correct photon trajectory calculation
    'TargetScoringPlaneHits': {
        'scalars':[],
        'vectors':['pdgID_', 'layerID_', 'x_', 'y_', 'z_',
                   'px_', 'py_', 'pz_', 'energy_']
    },
    'EcalVeto': {
        'scalars':['passesVeto_', 'nReadoutHits_', 'summedDet_',
                   'summedTightIso_', 'discValue_',
                   'recoilX_', 'recoilY_',
                   'recoilPx_', 'recoilPy_', 'recoilPz_'],
        'vectors':[]
    },
    'EcalRecHits': {
        'scalars':[],
        'vectors':['xpos_', 'ypos_', 'zpos_', 'energy_']  # OLD: ['id_', 'energy_']
    }
}

def blname(branch, leaf,sig):
    if sig:
        if branch.startswith('EcalVeto'):
            return '{}_signal/{}'.format(branch, leaf)
        else:
            return '{}_signal/{}_signal.{}'.format(branch, branch, leaf)
    else:
        if branch.startswith('EcalVeto'):
            return '{}_sim/{}'.format(branch, leaf)
        else:
            return '{}_sim/{}_sim.{}'.format(branch, branch, leaf)


def processFile(input_vars):
    # input_vars is a list:
    # [file_to_read, signal_mass, nth_file_in_mass_group]
    filename = input_vars[0]  # Apparently this is the easiest approach to multiple args...
    mass = input_vars[1]
    filenum = input_vars[2]
    sig = True
    if not mass:
        sig = False

    print("Processing file {}".format(filename))
    if mass == 0:
        outfile_name = "v14_pn_trigger_{}.root".format(filenum)
    else:
        outfile_name = "v14_{}_trigger_{}.root".format(mass, filenum)
    outfile_path = os.sep.join([output_dir, outfile_name])

    # NOTE:  Added this to ...
    if os.path.exists(outfile_path):
        print("FILE {} ALREADY EXISTS.  SKIPPING...".format(outfile_name))
        return 0, 0

    # Fix branch names:  uproot refers to EcalVeto branches with a / ('EcalVeto_v12/nReadoutHits_', etc), while
    # all other branches are referred to with a . ('EcalRecHits_v12.energy_', etc).  This is because ldmx-sw
    # writes EcalVeto information to the ROOT files in a somewhat unusual way; this may change in future updates
    # to ldmx-sw.
    branchList = []
    for branchname, leafdict in data_to_save.items():
        if sig:
            branchname_ = f'{branchname}_signal'
        else:
            branchname_ = f'{branchname}_sim'
        for leaf in leafdict['scalars'] + leafdict['vectors']:
            # EcalVeto needs slightly different syntax:   . -> /
            if branchname == "EcalVeto":
                branchList.append(branchname_ + '/' + leaf)
            else:
                branchList.append(branchname_ + '/' + branchname_+ '.' + leaf)

    print("Branches to load:")
    print(branchList)

    # Open the file and read all necessary data from it:
    t = uproot.open(filename)['LDMX_Events']
    print("OPENED FiLE")
    # (This part is just for printing the # of pre-preselection events:)
    #tmp = t.arrays(['EcalRecHits_v12/id_'])
    #nTotalEvents = len(tmp)
    #print("Before preselection:  found {} events".format(nTotalEvents))

    # t.arrays() returns a dict-like object:
    #    raw_data['EcalVeto_v12/nReadoutHits_'] == awkward array containing the value of 
    #    nReadoutHits_ for each event, and so on.
    raw_data = t.arrays(branchList) #, preselection)  #, aliases=alias_dict)
    print("Check raw_data:")
    print(raw_data[blname('EcalScoringPlaneHits','pdgID_',sig)])


    nTotalEvents = len(raw_data[blname('EcalRecHits', 'xpos_',sig)])
    print("Before preselection:  found {} events".format(nTotalEvents))

    # Perform the preselection:  Drop all events with more than MAX_NUM_ECAL_HITS in the ecal, 
    # and all events with an isolated energy that exceeds MAXX_ISO_ENERGY
    el = (raw_data[blname('EcalVeto', 'nReadoutHits_',sig)] < MAX_NUM_ECAL_HITS) * (raw_data[blname('EcalVeto', 'summedTightIso_',sig)] < MAX_ISO_ENERGY)
    preselected_data = {}
    for branch in branchList:
        preselected_data[branch] = raw_data[branch][el]
    nEvents = len(preselected_data[blname('EcalVeto', 'summedTightIso_',sig)])
    print("After preselection:  found {} events".format(nEvents))

    # Next, we have to compute TargetSPRecoilE_pt here instead of in train.py.  (This involves TargetScoringPlane
    # information that ParticleNet doesn't need, and that would take a long time to load with the lazy-loading
    # approach.)
    # For each event, find the recoil electron (maximal recoil pz):
    pdgID_ = t[blname('TargetScoringPlaneHits', 'pdgID_',sig)].array()[el]
    z_     = t[blname('TargetScoringPlaneHits', 'z_',sig)].array()[el]
    px_    = t[blname('TargetScoringPlaneHits', 'px_',sig)].array()[el]
    py_    = t[blname('TargetScoringPlaneHits', 'py_',sig)].array()[el]
    pz_    = t[blname('TargetScoringPlaneHits', 'pz_',sig)].array()[el]
    tspRecoil = []
    for i in range(nEvents):
        max_pz = 0
        recoil_index = 0  # Store the index of the recoil electron
        for j in range(len(pdgID_[i])):
            # Constraint on z ensures that the SP downstream of the target is used
            if pdgID_[i][j] == 11 and z_[i][j] > 0.176 and z_[i][j] < 0.178 and pz_[i][j] > max_pz:
                max_pz = pz_[i][j]
                recoil_index = j
        # Calculate the recoil SP
        if max_pz > 0:
            tspRecoil.append(np.sqrt(px_[i][recoil_index]**2 + py_[i][recoil_index]**2))
        else:
            tspRecoil.append(-999)
    # Put it in the preselected_data and treat it as an ordinary branch from here on out
    preselected_data['TargetSPRecoilE_pt'] = np.array(tspRecoil)

    # Additionally, add new branches storing the length for vector data (number of SP hits, number of ecal hits):
    nSPHits = np.zeros(nEvents) #[]
    nTSPHits = np.zeros(nEvents)
    nRecHits = np.zeros(nEvents) #[]
    x_data = preselected_data[blname('EcalScoringPlaneHits','x_',sig)]
    xsp_data = preselected_data[blname('TargetScoringPlaneHits','x_',sig)]
    E_data = preselected_data[blname('EcalRecHits','energy_',sig)]

    for i in range(nEvents):
        # NOTE:  max num hits may exceed MAX_NUM...this is okay.
        nSPHits[i] = len(x_data[i])  #nSPHits.append(len(x_data[i]))
        nTSPHits[i] = len(xsp_data[i])
        nRecHits[i] = sum(E_data[i] > 0)  #nRecHits.append(sum(E_data[i] > 0))  #len(E_data[i])) # NOTE:  Must be number of hits with E>0, since there's some E=0 hits out there...
        if len(E_data[i]) == 0:
            # ****** print("0 len! nrh was {}, i={}".format(sum(E_data[i] > 0), i))
            nRecHits[i] = 0
    preselected_data['nSPHits']  = np.array(nSPHits)
    preselected_data['nTSPHits'] = np.array(nTSPHits)
    preselected_data['nRecHits'] = np.array(nRecHits)


    # Prepare the output tree+file:
    outfile = r.TFile(outfile_path, "RECREATE")
    tree = r.TTree("skimmed_events", "skimmed ldmx event data")
    # Everything in EcalSPHits is a vector; everything in EcalVetoProcessor is a scalar

    # For each branch, create an array to temporarily hold the data for each event:
    scalar_holders = {}  # Hold ecalVeto (scalar) information
    vector_holders = {}
    for branch in branchList:
        leaf = re.split(r'[./]', branch)[-1]  #Split at / or .
        # Find whether the branch stores scalar or vector data:
        datatype = None
        for br, brdict in data_to_save.items():
            #print(leaf)
            #print(brdict['scalars'], brdict['vectors'])
            if leaf in brdict['scalars']:
                datatype = 'scalar'
                continue
            elif leaf in brdict['vectors']:
                datatype = 'vector'
                continue
        assert(datatype == 'scalar' or datatype == 'vector')
        if datatype == 'scalar':  # If scalar, temp array has a length of 1
            scalar_holders[branch] = np.zeros((1), dtype='float32')
        else:  # If vector, temp array must have at least one element per hit
            # (liberally picked 2k)
            vector_holders[branch] = np.zeros((200000), dtype='float32')
    print("TEMP:  Scalar, vector holders keys:")
    print(scalar_holders.keys())
    print(vector_holders.keys())

    # Create new branches to store nSPHits, pT (necessary for tree creation)...
    scalar_holders['nSPHits'] = np.array([0], 'i')
    scalar_holders['nTSPHits'] = np.array([0], 'i')
    scalar_holders['nRecHits'] = np.array([0], 'i')
    scalar_holders['TargetSPRecoilE_pt'] = np.array([0], dtype='float32')
    branchList.append('nSPHits')
    branchList.append('nTSPHits')
    branchList.append('nRecHits')
    branchList.append('TargetSPRecoilE_pt')
    # Now, go through each branch name and a corresponding branch to the tree:
    for branch, var in scalar_holders.items():
        # Need to make sure that each var is stored as the correct type (floats, ints, etc):
        if branch == 'nSPHits' or branch == 'nTSPHits' or branch == 'nRecHits':
            branchname = branch
            dtype = 'I'
        elif branch == 'TargetSPRecoilE_pt':
            branchname = branch
            dtype = 'F'
        else:
            branchname = re.split(r'[./]', branch)[1]
            dtype = 'F'
        tree.Branch(branchname, var, branchname+"/"+dtype)
    for branch, var in vector_holders.items():
        # NOTE:  Can't currently handle EcalVeto branches that store vectors.  Not necessary for PN, though.
        parent = re.split(r'[./]', branch)[0]
        branchname = re.split(r'[./]', branch)[-1]
        print("Found parent={}, branchname={}".format(parent, branchname))
        if parent == 'EcalScoringPlaneHits_signal'or parent == 'EcalScoringPlaneHits_sim':
            tree.Branch(branchname, var, "{}[nSPHits]/F".format(branchname))
        elif parent == 'TargetScoringPlaneHits_signal' or parent == 'TargetScoringPlaneHits_sim':
            tree.Branch(branchname+'tsp_', var, "{}[nTSPHits]/F".format(branchname+'tsp_'))
        else:  # else in EcalRecHits
            tree.Branch(branchname+'rec_', var, "{}[nRecHits]/F".format(branchname+'rec_'))
    print("TEMP:  Branches added to tree:")
    for b in tree.GetListOfBranches():  print(b.GetFullName())
    print("TEMP:  Leaves added ot tree:")
    for b in tree.GetListOfLeaves():    print(b.GetFullName())

    print("All branches added.  Filling...")

    for i in range(nEvents):
        # For each event, fill the temporary arrays with data, then write them to the tree with Fill()
        # ALSO:  If event contains no ecal hits, ignore it.
        if preselected_data['nRecHits'][i] == 0:  continue

        for branch in branchList:
            # Contains both vector and scalar data.  Treat them differently:
            if branch in scalar_holders.keys():  # Scalar
                # fill scalar data
                if i==0:  print("filling scalar", branch)
                scalar_holders[branch][0] = preselected_data[branch][i]
            elif branch in vector_holders.keys():  # Vector
                # fill vector data
                #print("vec data i is {}".format(type(preselected_data[branch][i])))
                #if len(preselected_data[branch][i] == 0):
                #    print("WARNING: found 0-len data, branch={}, i={}, nrh={}".format(branch, i, preselected_data['nRecHits'][i]))
                #    if preselected_data['nRecHits'][i] == 0:  print("  IS 0")

                #print(i, preselected_data[branch][i][0])  # make sure len is 1
                if i==0:  print("filling vector", branch)
                for j in range(len(preselected_data[branch][i])):
                    vector_holders[branch][j] = preselected_data[branch][i][j]
            else:
                print("FATAL ERROR:  {} not found in *_holders".format(branch))
                assert(False)
        tree.Fill()

    # Finally, write the filled tree to the ouput file:
    outfile.Write()
    print("FINISHED.  File written to {}.".format(outfile_path))

    return (nTotalEvents, nEvents)


if __name__ == '__main__':
    # New approach:  Use multiprocessing
    #pool = Pool(16) -> Run 16 threads/process 16 files in parallel
    
    presel_eff = {}
    # For each signal mass and for PN background:
    for mass, filepath in file_templates.items():
        print("======  m={}  ======".format(mass))
        # Assemble list of function params
        # These get passed to processFile() when Pool requests them
        params = []
        for filenum, f in enumerate(glob.glob(filepath)):
            params.append([f, mass, filenum])  # list will be passed to ProcessFile:  processFile([filepath, mass, file_number])
        with Pool(20) as pool:  # Can increase this number if desired, although this depends on how many threads POD will let you run at once...
            # this number is unclear, but 20 seems right judging from the POD webpage
            results = pool.map(processFile, params)
        print("Finished.  Result len:", len(results))
        print(results)
        nTotal  = sum([r[0] for r in results])
        nEvents = sum([r[1] for r in results])
        print("m = {} MeV:  Read {} events, {} passed preselection".format(int(mass*1000), nTotal, nEvents))
        if nTotal > 0:
            presel_eff[int(mass * 1000)] = float(nEvents) / nTotal
        else:
            presel_eff[int(mass * 1000)] = 'no events!'
    print("Done.  Presel_eff: {}".format(presel_eff))

    # For running without multithreading (note:  will be extremely slow and is impractical unless you want to test/use 1-2 files at a time):
    """
    presel_eff = {}
    for mass, filepath in file_templates.items():
        #if mass != 0:  continue
        filenum = 0
        nTotal = 0  # pre-preselection
        nEvents = 0 # post-preselection
        print("======  m={}  ======".format(mass))
        for f in glob.glob(filepath):
            # Process each file separately
            nT, nE = processFile([f, mass, filenum])
            nTotal += nT
            nEvents += nE
            filenum += 1
        print("m = {} MeV:  Read {} events, {} passed preselection".format(int(mass*1000), nTotal, nEvents))
        presel_eff[int(mass * 1000)] = nEvents / nTotal

    print("DONE.  presel_eff: ", presel_eff)
    """



