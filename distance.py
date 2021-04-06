import MDAnalysis as mda
import numpy as np
from numpy.linalg import norm
import pandas as pd


def molDistance(u,SOL_atoms,start,skip):
	# returns a dataframe
	
	# calculate Euclidean distance of its center of mass
	# from a reference frame
	
	# skip      : skip frames in the trajectory
	# start     : the index of starting frame
	# SOL_atoms : atom group of the individual solvent molecule

	distance = np.zeros(len(u.trajectory[start::skip]))
	time     = np.zeros(len(u.trajectory[start::skip]))
	for indx, ts in enumerate(u.trajectory[start::skip]):
	    # get the reference from the first frame of the trajectory
	    # (frames start from 0)
	    if ts.frame == start:
	        ref = SOL_atoms.center_of_mass()
	        
	    # calculate distance and 
	    distance[indx] = norm( SOL_atoms.center_of_mass() - ref )
	    time[indx]     = ts.time

	# ps to ns
	time     = time*1.0e-3
	# angstrom to nm
	distance = distance*1.0e-1
	
	return pd.DataFrame({'time': time, 'COM_L2_nm': distance})

def filterDistance(df,threshold,skip):
    # returns numpy array
    df  = df[ df['COM_L2_nm'] <= threshold ]
    dis = df['COM_L2_nm'].to_numpy()
    return dis[0::skip]

def getX(bins):
    X = []
    for i in range(1,len(bins)):
        X.append(0.5*(bins[i-1] + bins[i]))
    return np.array(X)

def PMF(dis,nbins):
    # PMF: Potential Mean Force
    # returns two numpy arrays
    dis          = np.delete(dis,0) # remove the first element because it is zero
    counts, bins = np.histogram(dis,density=True,bins=nbins)
    rc           = getX(bins)
    # clear counts and bins when zero is encountered
    data    = {'rc': rc, 'density': counts}
    df      = pd.DataFrame(data)
    df      = df[ df['density'] != 0.0 ]
    rc      = df['rc'].to_numpy()
    density = df['density'].to_numpy()
    fes     = -np.log(density)
    return rc, fes