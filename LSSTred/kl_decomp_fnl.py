import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import erf
import pyccl as ccl
from scipy.integrate import quad
import matplotlib.cm as cm
import common_gofish as cgf

plot_stuff=False
SZ_RED=0.02
LMAX=500
nsamp=1
ZMAX=2
tracertype='gal_clustering'
parname='fnl'
par0=0.0
dpar=0.5
sigma_gamma=1.
zedge_lo=0.5
zedge_hi=1.4
prefix="fnl"
nzfile='nz_red.txt'

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
xcorr=np.array([[np.sum(nz1*nz2) for nz1 in nz_bins] for nz2 in nz_bins])
xcorr/=np.sqrt(np.diag(xcorr)[:,None]*np.diag(xcorr)[None,:])

#Compute number densities
ndens=np.array([np.sum(nz)*(zarr[1]-zarr[0]) for nz in nz_bins])
print quad(nz_red,0,5)[0],np.sum(nzarr*(zarr[1]-zarr[0])),np.sum(ndens)

run_name="sz%.2lf_"%SZ_RED+"ns%d_"%nsamp+"lmx%d"%LMAX
np.savetxt("bins_"+run_name+".txt",np.transpose([z0bins,zfbins,sz_red(0.5*(z0bins+zfbins))]),fmt='%lf %lf %lf 0 0 '+'%d'%LMAX,header='[1]-z0 [2]-zf [3]-sz [4]-marg_sz [5]-marg_bz [6]-lmax')
zb_arr=np.linspace(0,ZMAX,16); bb_arr=bz_red(zb_arr); np.savetxt("bz_"+run_name+".txt",np.transpose([zb_arr,bb_arr]),fmt='%lf %lf 0');
np.savetxt("nz_"+run_name+".txt",np.transpose([zarr,nzarr]),fmt='%lf %lf')

#Plot N(z)
if plot_stuff :
    plt.figure();
    for i in nbins-1-np.arange(nbins) :
        nz=nz_bins[i]
        ran=np.where(nz>1E-3*np.amax(nz))[0]
        plt.plot(zarr[ran],nz[ran],color=cm.brg((nbins-i-0.5)/nbins),lw=2)
    plt.plot(zarr,nzarr,'k-',lw=2)
    plt.xlim([0,1.1*np.amax(nzarr)])
    plt.xlabel('$z$',fontsize=18)
    plt.ylabel('$N(z)\\,\\,[{\\rm arcmin}^{-2}]$',fontsize=18)

#Compute power spectra
c_ij_fid,c_ij_mfn,c_ij_pfn=cgf.run_gofish(run_name,LMAX,parname,par0,dpar,tracertype)
n_ij_fid=np.zeros_like(c_ij_fid)
for i1 in np.arange(nbins) :
    n_ij_fid[:,i1,i1]=sigma_gamma**2*(np.pi/180./60.)**2/ndens[i1]
    c_ij_fid[:,i1,i1]+=n_ij_fid[:,i1,i1]
c_ij_dfn=(c_ij_pfn-c_ij_mfn)/(2*dpar)
larr=np.arange(LMAX+1)
inv_cij=np.linalg.inv(c_ij_fid)
metric=np.linalg.inv(n_ij_fid)
dprod=np.array([np.dot(c_ij_dfn[l,:,:],inv_cij[l,:,:]) for l in np.arange(LMAX+1)])
fish=(larr+0.5)*np.array([np.trace(np.dot(d,d)) for d in dprod])
sigma=np.sqrt(1./np.sum(fish))
print sigma

#Plot uncoupled power spectra
if plot_stuff :
    plt.figure();
    for i1 in np.arange(nbins) :
        col=cm.brg((nbins-i1-0.5)/nbins)
        plt.plot(larr,(c_ij_fid[:,i1,i1]-n_ij_fid[:,i1,i1]),'-',color=col,lw=2)
        plt.plot(larr,(n_ij_fid[:,i1,i1]                  ),'--',color=col,lw=1)
    plt.loglog()
    plt.xlabel('$\\ell$',fontsize=18)
    plt.ylabel('$C^{\\alpha\\beta}_\\ell$',fontsize=18)


def change_basis(c,m,ev) :
    print np.shape(c), np.shape(m), np.shape(ev)
    return np.array([np.diag(np.dot(np.transpose(ev[l]),np.dot(m[l],np.dot(c[l],np.dot(m[l],ev[l])))))
                     for l in np.arange(LMAX+1)])

def diagonalize(c,m) :
    im=np.linalg.inv(m)
    ll=np.linalg.cholesky(m)
    ill=np.linalg.cholesky(im)
    cl=np.array([np.dot(np.transpose(ll[l]),np.dot(c[l],ll[l])) for l in np.arange(LMAX+1)])
    c_p,v=np.linalg.eigh(cl)
    ev=np.array([np.dot(np.transpose(ill[l]),v[l]) for l in np.arange(LMAX+1)])

    iden=change_basis(im,m,ev)
    return ev,c_p

#Get K-L modes
e_v,c_p_fid=diagonalize(c_ij_fid,np.linalg.inv(n_ij_fid))
c_p_dfn  =change_basis(c_ij_dfn,metric,e_v)
fisher=(larr+0.5)[:,None]*(c_p_dfn/c_p_fid)**2
isort=np.argsort(-np.sum(fisher,axis=0))
e_o=e_v[:,:,isort]
c_p_fid=change_basis(c_ij_fid,metric,e_o)
c_p_dfn=change_basis(c_ij_dfn,metric,e_o)

#Plot power spectrum of K-L modes
if plot_stuff :
    plt.figure();
    ax=plt.gca()
#ax.imshow([[0.,1.],[0.,1.]],extent=[700,1600,140,200],interpolation='bicubic',cmap=cm.summer,aspect='auto')
#plt.text(220,160,'$p\\in[1,16]$',{'fontsize':16})
    for i in np.arange(nbins) :
        c=c_p_fid[:,i]
        col=cm.brg((i+0.5)/nbins)
        plt.plot(larr,c,color=col,lw=2)
    plt.ylabel("$D_\\ell^p$",fontsize=16)
    plt.xlabel("$\\ell$",fontsize=16)
    plt.loglog()


#Plot K-L eigenvectors
if plot_stuff :
    plt.figure()
    ax=plt.gca()
    zbarr=0.5*(z0bins+zfbins)
#ax.imshow([[0.,1.],[0.,1.]],extent=[2.0,2.24,-0.35,-0.30],interpolation='bicubic',cmap=cm.winter,aspect='auto')
#ax.imshow([[0.,1.],[0.,1.]],extent=[2.0,2.24,-0.42,-0.37],interpolation='bicubic',cmap=cm.autumn,aspect='auto')
#plt.text(1.7,-0.34,'$1^{\\rm st}\\,\\,{\\rm mode}$',{'fontsize':16})
#plt.text(1.7,-0.41,'$2^{\\rm nd}\\,\\,{\\rm mode}$',{'fontsize':16})
    for i in (1+np.arange(9))*10 :
        ax.plot(zbarr,e_o[  i,:,0]*np.sqrt(ndens)/np.sqrt(np.sum(e_o[  i,:,0]**2*ndens)),'o-',markeredgewidth=0,
             color=cm.winter((i+0.5)/101))
        if e_o[i,2,1]>0 :
            sign=-1
        else :
            sign=1
        ax.plot(zbarr,sign*e_o[  i,:,1]*np.sqrt(ndens)/np.sqrt(np.sum(e_o[  i,:,1]**2*ndens)),'o-',
                markeredgewidth=0,color=cm.autumn((i+0.5)/101))
    plt.xlabel('$z_\\alpha$',fontsize=18)
    plt.ylabel('$\\sqrt{\\bar{n}^\\alpha}\\,({\\sf E}_\\ell)^1_\\alpha$',fontsize=18)


fisher=(larr+0.5)[:,None]*(c_p_dfn/c_p_fid)**2
fish_permode=np.sum(fisher,axis=0)
fish_cum=np.cumsum(fish_permode)

if plot_stuff :
    plt.figure();
    imodes=np.arange(nbins)+1
    plt.plot(imodes,fish_permode/np.sum(fish_permode),'go-',lw=2,label='${\\rm Information\\,\\,in\\,\\,mode}\\,\\,p_{\\rm KL}$',markeredgewidth=0)
    plt.plot(imodes[:-1],1-fish_cum[:-1]/fish_cum[-1],'ro-',lw=2,label='${\\rm Information\\,\\,in\\,\\,modes}\\,\\,>p_{\\rm KL}$',markeredgewidth=0)
    plt.legend(loc='upper right',frameon=False)
    plt.xlabel('${\\rm KL\\,\\,mode\\,\\,order}\\,\\,p_{\\rm KL}$',fontsize=18)
    plt.ylabel('${\\rm Relative\\,information\\,\\,content}$',fontsize=18)
    plt.yscale('log')
    plt.xlim([0.9,nbins+0.1])
    plt.ylim([3E-5,1.2])

if plot_stuff :
    plt.show()
