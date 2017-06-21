import numpy as np
from scipy.interpolate import interp1d

#z,az,dum=np.loadtxt("az_gold.txt",unpack=True)
zarr=4*np.arange(512)/511.
#azf=interp1d(z,az)
#azarr=azf(zarr)
azarr=np.ones_like(zarr)
rfarr=np.ones_like(zarr)
#rfarr=zarr*np.exp(-(zarr/1.5)**1.8)/4.5; print np.amax(rfarr)
np.savetxt("az.txt",np.transpose([zarr,azarr]))
np.savetxt("rf.txt",np.transpose([zarr,rfarr]))
