import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import erf
import pyccl as ccl
from scipy.integrate import quad
import matplotlib.cm as cm
import common_gofish as cgf

plot_stuff=True
SZ_RED=0.05
LMAX=2000
nsamp=1
ZMAX=3
tracertype='gal_shear'
parname='w0'
par0=-1.0
dpar=0.05
sigma_gamma=0.28
zedge_lo=0.5
zedge_hi=2.5
prefix="wl0"
nzfile='nz_blue.txt'

def sz_red(z) :
    return SZ_RED*(1+z)

def bz_red(z) :
    return 1+z

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
nzarr=nz_red(zarr)

#Selection function for individual bins
z0bins,zfbins=get_edges(zedge_lo,zedge_hi,SZ_RED,nsamp)
nbins=len(z0bins)
nz_bins=np.array([nzarr*pdf_photo(zarr,z0,zf,sz_red(0.5*(z0+zf))) for z0,zf in zip(z0bins,zfbins)])
ndens=np.array([np.sum(nz)*(zarr[1]-zarr[0]) for nz in nz_bins])

data_ll=cgf.read_cls_class("outputs_wl0_IA_sz0.05_ns1_lmx2000/run_lens_lenscl.dat")
data_ii=cgf.read_cls_class("outputs_wl0_IA_sz0.05_ns1_lmx2000/run_ia_iacl.dat")
data_li=cgf.read_cls_class("outputs_wl0_IA_sz0.05_ns1_lmx2000/run_lens_iacl.dat")

larr=np.arange(2001)
cl_ll=data_ll['cl_ll']
cl_ii=data_ii['cl_ll']
cl_li=data_li['cl_ll'][:,16:,:16]
cl_il=data_li['cl_ll'][:,:16,16:]

c_ij_fid=cl_ll+cl_li+cl_il+cl_ii
n_ij_fid=np.zeros_like(c_ij_fid)
for i1 in np.arange(nbins) :
    n_ij_fid[:,i1,i1]=sigma_gamma**2*(np.pi/180./60.)**2/ndens[i1]
    c_ij_fid[:,i1,i1]+=n_ij_fid[:,i1,i1]
n_ij_fid+=cl_ii+(cl_il+cl_li)*0.5
larr=np.arange(LMAX+1)

c_ij_fid=c_ij_fid[100:,:,:]
n_ij_fid=n_ij_fid[100:,:,:]
larr=larr[100:]
nell=len(larr)

#Compute power spectra
inv_cij=np.linalg.inv(c_ij_fid)
metric=np.linalg.inv(n_ij_fid)
#dprod=np.array([np.dot(c_ij_dfn[l,:,:],inv_cij[l,:,:]) for l in np.arange(LMAX+1)])
#fish=(larr+0.5)*np.array([np.trace(np.dot(d,d)) for d in dprod])
#sigma=np.sqrt(1./np.sum(fish))

def change_basis(c,m,ev) :
    return np.array([np.diag(np.dot(np.transpose(ev[l]),np.dot(m[l],np.dot(c[l],np.dot(m[l],ev[l])))))
                     for l in np.arange(nell)])

def diagonalize(c,m) :
    w,v=np.linalg.eigh(m)
    print np.shape(w)
    print np.shape(v)
    print np.amin(w,axis=1),np.amax(w,axis=1)
#    print np.amax(w,axis=0),np.amin(w,axis=0)
    plt.plot(np.amin(w,axis=1)); plt.show()
    im=np.linalg.inv(m)
    ll=np.linalg.cholesky(m)
    ill=np.linalg.cholesky(im)
    cl=np.array([np.dot(np.transpose(ll[l]),np.dot(c[l],ll[l])) for l in np.arange(nell)])
    c_p,v=np.linalg.eigh(cl)
    ev=np.array([np.dot(np.transpose(ill[l]),v[l]) for l in np.arange(nell)])

    iden=change_basis(im,m,ev)
    return ev,c_p

#Get K-L modes
e_v,c_p_fid=diagonalize(-c_ij_fid,metric)
e_o=e_v
c_p_fid=change_basis(c_ij_fid,metric,e_o)

#Plot power spectrum of K-L modes
if plot_stuff :
    plt.figure();
    ax=plt.gca()
    ax.imshow([[0.,1.],[0.,1.]],extent=[700,1600,140,200],interpolation='bicubic',cmap=cm.summer,aspect='auto')
    plt.text(220,160,'$p\\in[1,16]$',{'fontsize':16})
    for i in np.arange(nbins) :
        c=c_p_fid[:,i]
        col=cm.summer((i+0.5)/nbins)
        plt.plot(larr,c,color=col,lw=2)
    plt.ylabel("$D_\\ell^p$",fontsize=16)
    plt.xlabel("$\\ell$",fontsize=16)
    plt.xlim([100,2000])
    plt.ylim([0.5,300])
    plt.loglog()
#    plt.savefig('../Draft/Figs/d_p_wl.pdf',bbox_inches='tight')


#Plot K-L eigenvectors
if plot_stuff :
    plt.figure()
    ax=plt.gca()
    zbarr=0.5*(z0bins+zfbins)
    ax.plot([0.5,2.3],[0,0],'k--')
    ax.imshow([[0.,1.],[0.,1.]],extent=[2.0,2.24,-0.35,-0.30],interpolation='bicubic',cmap=cm.winter,aspect='auto')
    ax.imshow([[0.,1.],[0.,1.]],extent=[2.0,2.24,-0.42,-0.37],interpolation='bicubic',cmap=cm.autumn,aspect='auto')
    plt.text(1.345,-0.335,'$1^{\\rm st}\\,\\,{\\rm mode},\\,\\,\\ell\\in[2,2000]$',{'fontsize':16})
    plt.text(1.33 ,-0.41 ,'$2^{\\rm nd}\\,\\,{\\rm mode},\\,\\,\\ell\\in[2,2000]$',{'fontsize':16})
    for i in (1+np.arange(189))*10 :
        ax.plot(zbarr,-e_o[  i,:,0]*np.sqrt(ndens)/np.sqrt(np.sum(e_o[  i,:,0]**2*ndens)),'o-',
                markeredgewidth=0,color=cm.winter((i+0.5)/2001))
        if e_o[i,2,1]>0 :
            sign=1
        else :
            sign=-1
        ax.plot(zbarr,sign*e_o[  i,:,1]*np.sqrt(ndens)/np.sqrt(np.sum(e_o[  i,:,1]**2*ndens)),'o-',
                markeredgewidth=0,color=cm.autumn((i+0.5)/2001))
    plt.xlabel('$z_\\alpha$',fontsize=18)
    plt.ylabel('$\\sqrt{\\bar{n}^\\alpha}\\,({\\sf E}_\\ell)^1_\\alpha$',fontsize=18)
    plt.xlim([0.5,2.3])
#    plt.savefig('../Draft/Figs/kl_modes_wl.pdf',bbox_inches='tight')
plt.show()
