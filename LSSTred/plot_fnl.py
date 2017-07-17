import numpy as np
import matplotlib.pyplot as plt

data_kl  =np.loadtxt("sigmas_kl.txt"  ,unpack=True)
data_tm  =np.loadtxt("sigmas_tm.txt"  ,unpack=True)
data_klsn=np.loadtxt("sigmas_klsn.txt",unpack=True)
plt.plot(data_kl[0],data_kl[1]/data_kl[1,-1]-1,'ro-',lw=2,label='${\\rm K-L\\,\\,decomp.\\,\\,for\\,\\,}f_{\\rm NL}$',markeredgewidth=0)
plt.plot(data_tm[0],data_tm[1]/data_tm[1,-1]-1,'bo-',lw=2,label='${\\rm Tomography}$',markeredgewidth=0)
plt.plot(data_klsn[0],data_klsn[1]/data_klsn[1,-1]-1,'o-',color='#AAAAAA',lw=2,label='${\\rm K-L\\,\\,decomp.\\,\\,for\\,\\,}S/N$',markeredgewidth=0)
plt.legend(loc='lower left',frameon=False,fontsize=18)
plt.xlabel('${\\rm Number\\,\\,of\\,\\,modes}$',fontsize=18)
plt.ylabel('$\\Delta\\sigma(f_{\\rm NL})/\\sigma_{\\rm best}(f_{\\rm NL})$',fontsize=18)
plt.xlim([0.9,14.1])
plt.ylim([2E-3,10])
plt.yscale('log')
plt.savefig("../Draft/Figs/kl_fnl.pdf",bbox_inches='tight')
plt.show()
