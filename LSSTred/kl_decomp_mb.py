import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import erf
#import pyccl as ccl
from scipy.integrate import quad
import matplotlib.cm as cm
import common_gofish as cgf
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

plot_stuff=True
SZ_RED=0.05
LMAX=2000
nsamp=1
ZMAX=4
sigma_gamma=0.28
zedge_lo=0.1
zedge_hi=2.5
nzfile='nz_blue.txt'

def sphz_red(z) :
    return SZ_RED*(1+z)

def bz_red(z) :
    return 1+0.8*z

def sz_red(z) :
    return np.ones_like(z)*0.6

def pdf_photo(z,z0,zf,sz) :
    denom=1./np.sqrt(2*sz*sz)
    return 0.5*(erf((zf-z)*denom)-erf((z0-z)*denom))

def get_edges(zmin,zmax,sz,frac_sigma) :
    zout=[]
    zout.append(zmin)
    dz0=frac_sigma*sz
    i=0
    while zout[i]<=zmax :
        i+=1
        zout.append((dz0+zout[i-1]*(1+dz0*0.5))/(1-0.5*dz0))
    zout=np.array(zout); zout=zout[zout<zmax]
    z0out=zout[:-1]; zfout=zout[1:]

    return z0out,zfout

#Selection function for the whole sample
data=np.loadtxt(nzfile,unpack=True)
nz_red=interp1d(data[0],data[1],bounds_error=False,fill_value=0)
zarr=np.linspace(0,ZMAX,1024)
bzarr=bz_red(zarr)
szarr=sz_red(zarr)
nzarr=nz_red(zarr)

np.savetxt("outputs_gc_mb/bz.txt",np.transpose([zarr,bzarr]))
np.savetxt("outputs_gc_mb/sz.txt",np.transpose([zarr,szarr]))
np.savetxt("outputs_gc_mb/nz.txt",np.transpose([zarr,nzarr]))

#Selection function for individual bins
z0bins,zfbins=get_edges(zedge_lo,zedge_hi,SZ_RED,nsamp); zbarr=0.5*(z0bins+zfbins); nbins=len(z0bins)
nz_bins=np.array([nzarr*pdf_photo(zarr,z0,zf,sphz_red(0.5*(z0+zf))) for z0,zf in zip(z0bins,zfbins)])
print np.shape(nz_bins),np.shape(zarr)
ndens=np.array([np.sum(nz)*(zarr[1]-zarr[0]) for nz in nz_bins])
np.savetxt("outputs_gc_mb/bins.txt",
           np.transpose([0.5*(z0bins+zfbins),0.5*(zfbins-z0bins),sphz_red(0.5*(z0bins+zfbins))]))

ib_off=7
if plot_stuff :
    plt.figure();
    for i in nbins-1-np.arange(nbins) :
        if i>=ib_off :
            nz=nz_bins[i]
            ran=np.where(nz>1E-3*np.amax(nz))[0]
            plt.plot(zarr[ran],nz[ran],color=cm.brg((nbins-ib_off-i+ib_off-0.5)/(nbins-ib_off+0.)),lw=2)
    plt.plot(zarr,nzarr,'k-',lw=2)
    plt.ylim([0,1.05*np.amax(nzarr)])
    plt.xlim([0,1.2*zedge_hi])
    plt.xlabel('$z$',fontsize=18)
    plt.ylabel('$N(z)\\,\\,[{\\rm arcmin}^{-2}]$',fontsize=18)
    plt.savefig('../Draft/Figs/nz_lsst_mb.pdf',bbox_inches='tight')

data_dd=cgf.read_cls_class("outputs_gc_mb/run_dens_denscl.dat")
data_mm=cgf.read_cls_class("outputs_gc_mb/run_mb_mbcl.dat")
data_dm=cgf.read_cls_class("outputs_gc_mb/run_dens_mbcl.dat")

larr=np.arange(2001)[2:1001]
cl_dd=data_dd['cl_dd'][2:1001,ib_off:,ib_off:]
cl_mm=data_mm['cl_dd'][2:1001,ib_off:,ib_off:]
cl_dm=data_dm['cl_dd'][2:1001,nbins+ib_off:,ib_off:nbins]
cl_md=data_dm['cl_dd'][2:1001,ib_off:nbins,nbins+ib_off:]
nbins-=ib_off

c_ij_fid=(cl_dd+cl_dm+cl_md+cl_mm)
d_ij_fid=2*(cl_mm+0.5*(cl_dm+cl_md))
n_ij_fid=np.zeros_like(c_ij_fid)
for i1 in np.arange(nbins) :
    n_ij_fid[:,i1,i1]=(np.pi/180./60.)**2/ndens[i1+ib_off]
c_ij_fid+=n_ij_fid
nell=len(c_ij_fid)

if plot_stuff :
    c_m=(c_ij_fid-n_ij_fid)[400,:,:]
    c_d=cl_dd[400,:,:]
    r_m=c_m/np.sqrt(np.diag(c_m)[:,None]*np.diag(c_m)[None,:])
    r_d=c_d/np.sqrt(np.diag(c_d)[:,None]*np.diag(c_d)[None,:])

    plt.figure(); ax=plt.gca();
    ax.set_title('$R^{\\alpha,\\beta}_{\\ell=400},\\,\\,{\\rm w.\\,\\,magnification}$',fontsize=18)
    im=ax.imshow(r_m,origin='lower',extent=[1-0.5,nbins+0.5,1-0.5,nbins+0.5],
                 norm=LogNorm(vmin=0.01,vmax=1),interpolation='nearest',cmap=cm.bone)
    ax.set_xlabel('${\\rm Bin\\,\\,1}$',fontsize=18)
    ax.set_ylabel('${\\rm Bin\\,\\,2}$',fontsize=18)
    plt.colorbar(im)
    plt.savefig('../Draft/Figs/r_ij_mb_wm.pdf',bbox_inches='tight')

    plt.figure(); ax=plt.gca();
    ax.set_title('$R^{\\alpha,\\beta}_{\\ell=400},\\,\\,{\\rm w.o.\\,\\,magnification}$',fontsize=18)
    im=ax.imshow(np.fabs(r_d),origin='lower',extent=[1-0.5,nbins+0.5,1-0.5,nbins+0.5],
                 norm=LogNorm(vmin=0.01,vmax=1),interpolation='nearest',cmap=cm.bone)
    ax.set_xlabel('${\\rm Bin\\,\\,1}$',fontsize=18)
    ax.set_ylabel('${\\rm Bin\\,\\,2}$',fontsize=18)
    plt.colorbar(im)
    plt.savefig('../Draft/Figs/r_ij_mb_wom.pdf',bbox_inches='tight')

def diagonalize(a,f) :
    mm=np.linalg.cholesky(np.linalg.inv(f))
    ap=np.array([np.dot(np.transpose(mm[l]),np.dot(a[l],mm[l])) for l in np.arange(nell)])
    lam,e=np.linalg.eigh(-ap); lam*=-1
    v=np.array([np.dot(mm[l],e[l]) for l in np.arange(nell)])

    return v,lam

vv2,llam2=diagonalize(d_ij_fid,c_ij_fid)
isort=np.argsort(-np.sum((larr+0.5)[:,None]*llam2**2,axis=0))
vv=vv2[:,:,isort]
llam=np.array([np.diag(np.dot(np.transpose(vv[l]),np.dot(d_ij_fid[l],vv[l]))) for l in np.arange(nell)])

for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] :
    idn=np.ones(nbins); idn[i:]=0; p1=np.diag(idn); 
    proj=np.dot(c_ij_fid[100,:,:],np.dot(vv[100,:,:],np.dot(p1,np.transpose(vv[100,:,:]))))
    plt.figure(); plt.imshow(proj,origin='lower',interpolation='nearest',vmin=-0.1,vmax=1)
plt.show()
exit(1)
#p1=np.diag
print np.dot(np.transpose(vv[10,:,:]),np.dot(c_ij_fid[10,:,:],vv[10,:,:]))
print np.dot(np.dot(c_ij_fid[10,:,:],vv[10,:,:]),np.transpose(vv[10,:,:]))
exit(1)



fisher=(larr+0.5)[:,None]*llam**2
fish_permode=np.sum(fisher,axis=0)
fish_cum=np.cumsum(fish_permode)

if plot_stuff :
    plt.figure();
    imodes=np.arange(nbins)+1
    plt.plot(imodes,fish_permode/np.sum(fish_permode),'go-',lw=2,
             label='${\\rm Information\\,\\,in\\,\\,mode}\\,\\,p_{\\rm KL}$',markeredgewidth=0);
    plt.plot(imodes[:-1],1-fish_cum[:-1]/fish_cum[-1],'ro-',lw=2,
             label='${\\rm Information\\,\\,in\\,\\,modes}\\,\\,>p_{\\rm KL}$',markeredgewidth=0)
    plt.legend(loc='upper right',frameon=False)
    plt.xlabel('${\\rm KL\\,\\,mode\\,\\,order}\\,\\,p_{\\rm KL}$',fontsize=18)
    plt.ylabel('${\\rm Relative\\,information\\,\\,content}$',fontsize=18)
    plt.xlim([0.9,9.5])
    plt.ylim([-0.03,1.03])
    plt.savefig('../Draft/Figs/information_mb.pdf',bbox_inches='tight')

if plot_stuff :
    plt.show();

ch=np.linalg.cholesky(c_ij_fid)
ww=np.array([np.dot(np.transpose(ch[l]),vv[l]) for l in np.arange(nell)])

plt.figure()
for i in (1+np.arange(99))*10 :
    v=vv[i,:,0]; v/=np.sqrt(np.sum(v**2))
    sign=1
    if v[10]<0 :
        sign=-1
    plt.plot(zbarr[ib_off:],v*sign,'ro-')

    v=vv[i,:,1]; v/=np.sqrt(np.sum(v**2))
    sign=1
    if v[10]<0 :
        sign=-1
    plt.plot(zbarr[ib_off:],v*sign,'bo-')
plt.show()


exit(1)
