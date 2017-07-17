import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import erf
import pyccl as ccl
from scipy.integrate import quad
import matplotlib.cm as cm
import common_gofish as cgf
import os

plot_stuff=False
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

OC0=0.2667
OB0=0.0495
HH0=0.67
S80=0.84
NS0=0.96
OM0=OC0+OB0
print OM0
DOM=2E-3
NOM=64
NS8=64

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

run_name=prefix+"_sz%.2lf_"%SZ_RED+"ns%d_"%nsamp+"lmx%d"%LMAX
np.savetxt("bins_"+run_name+".txt",np.transpose([z0bins,zfbins,sz_red(0.5*(z0bins+zfbins))]),fmt='%lf %lf %lf 0 0 '+'%d'%LMAX,header='[1]-z0 [2]-zf [3]-sz [4]-marg_sz [5]-marg_bz [6]-lmax')
zb_arr=np.linspace(0,ZMAX,16); bb_arr=bz_red(zb_arr); np.savetxt("bz_"+run_name+".txt",np.transpose([zb_arr,bb_arr]),fmt='%lf %lf 0');
np.savetxt("nz_"+run_name+".txt",np.transpose([zarr,nzarr]),fmt='%lf %lf')

#Compute power spectra
if os.path.isfile("cl_ccl_fid.npy") :
    c_ij_fid=np.load("cl_ccl_fid.npy")
else :
    c_ij_fid=cgf.compute_cls(OC0,OB0,HH0,S80,NS0,par0     ,nbins,zarr,nz_bins,LMAX)
    np.save("cl_ccl_fid",c_ij_fid)
if os.path.isfile("cl_ccl_mfn.npy") :
    c_ij_mfn=np.load("cl_ccl_mfn.npy")
else :
    c_ij_mfn=cgf.compute_cls(OC0,OB0,HH0,S80,NS0,par0-dpar,nbins,zarr,nz_bins,LMAX)
    np.save("cl_ccl_mfn",c_ij_mfn)
if os.path.isfile("cl_ccl_pfn.npy") :
    c_ij_pfn=np.load("cl_ccl_pfn.npy")
else :
    c_ij_pfn=cgf.compute_cls(OC0,OB0,HH0,S80,NS0,par0+dpar,nbins,zarr,nz_bins,LMAX)
    np.save("cl_ccl_pfn",c_ij_pfn)
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

def change_basis(c,m,ev) :
    return np.array([np.diag(np.dot(np.transpose(ev[l]),np.dot(m[l],np.dot(c[l],np.dot(m[l],ev[l])))))
                     for l in np.arange(LMAX+1)])

def diagonalize(c,m) :
    im=np.linalg.inv(m)
    ll=np.linalg.cholesky(m)
    ill=np.linalg.cholesky(im)
    cl=np.array([np.dot(np.transpose(ll[l]),np.dot(c[l],ll[l])) for l in np.arange(LMAX+1)])
    c_p,v=np.linalg.eigh(cl)
    ev=np.array([np.dot(np.transpose(ill[l]),v[l]) for l in np.arange(LMAX+1)])
    bv=np.array([np.dot(np.transpose(ll[l]),v[l]) for l in np.arange(LMAX+1)])

    iden=change_basis(im,m,ev)
    return ev,bv,c_p

#Get K-L modes
e_v,b_v,c_p_fid=diagonalize(c_ij_fid,metric)
c_p_dfn  =change_basis(c_ij_dfn,metric,e_v)
fisher=(larr+0.5)[:,None]*(c_p_dfn/c_p_fid)**2
isort=np.argsort(-np.sum(fisher,axis=0))
e_o=e_v[:,:,isort]
c_p_fid=change_basis(c_ij_fid,metric,e_o)
c_p_dfn=change_basis(c_ij_dfn,metric,e_o)

def gen_random_real(cll,ev,ng,csig,ctot) :
    lmax=len(cll)-1.
    lbox=(np.pi*ng/lmax)
    nfields=len(cll[0])
    print lbox*180/np.pi
    larr=np.arange(len(cll))

    filter_ell=np.zeros_like(cll)
    filter_dc_ell=np.maximum(cll-1,0)/cll
    for i in np.arange(nfields) :
        filter_ell[:,i]=csig[:,i,i]/ctot[:,i,i]
        
    ell1d=np.fft.fftfreq(ng)*2*np.pi*ng/lbox
    ellx2d=ell1d[None,:]*(np.ones_like(ell1d))[:,None]
    elly2d=ell1d[:,None]*(np.ones_like(ell1d))[None,:]
    ell2d=np.sqrt(ellx2d**2+elly2d**2)

    ids=np.minimum(ell2d.astype(int),int(lmax))
    sigmas=np.sqrt(cll[ids,:]*(lbox*0.5/np.pi)**2*0.5)
    filters=filter_ell[ids,:]
    filters_dc=filter_dc_ell[ids,:]

    #Decoupled field, Fourier space
    field_fourier_decoupled=(np.random.randn(ng,ng,nfields)+1j*np.random.randn(ng,ng,nfields))*sigmas

    def transf_c2r(c) :
        return np.real(np.fft.fft2(c)*2*np.pi/lbox**2)*np.sqrt(2.)

    def transf_r2c(r) :
        return np.fft.ifft2(r)*lbox**2/(2*np.pi)

    def get_pk_r(r1,r2) :
        c1=transf_r2c(r1)
        c2=transf_r2c(r2)
        p=(np.real(c1)*np.real(c2)+np.imag(c1)*np.imag(c2))*(2*np.pi/lbox)**2
        hn,b=np.histogram(ell2d,range=[0,lmax],bins=ng/2)
        hp,b=np.histogram(ell2d,range=[0,lmax],bins=ng/2,weights=p)
        ll=0.5*(b[1:]+b[:-1])
        pp=hp/hn
        return ll,pp

    def get_pk_c(c1,c2) :
        p=(np.real(c1)*np.real(c2)+np.imag(c1)*np.imag(c2))*(2*np.pi/lbox)**2
        hn,b=np.histogram(ell2d,range=[0,lmax],bins=ng/2)
        hp,b=np.histogram(ell2d,range=[0,lmax],bins=ng/2,weights=p)
        ll=0.5*(b[1:]+b[:-1])
        pp=hp/hn
        return ll,pp

    #Couple fields, Fourier space
    bm1=ev
    bm1mp=bm1[ids,:,:]
    field_fourier_coupled=np.zeros_like(field_fourier_decoupled)
    for iy in np.arange(ng) :
        for ix in np.arange(ng) :
            field_fourier_coupled[iy,ix,:]=np.dot(bm1mp[iy,ix,:,:],field_fourier_decoupled[iy,ix,:])

    #Transform fields to real space
    field_real_decoupled=np.zeros([ng,ng,nfields])
    field_real_coupled=np.zeros([ng,ng,nfields])
    for i in np.arange(nfields) :
        field_real_decoupled[:,:,i]=transf_c2r(field_fourier_decoupled[:,:,i])
        field_real_coupled[:,:,i]  =transf_c2r(field_fourier_coupled[:,:,i])
       
    plt.figure()
    for i in np.arange(1) :
        for j in np.arange(nfields-i)+i :
            lrr,prrij=get_pk_r(field_real_coupled[:,:,i],field_real_coupled[:,:,j])
            lrr,prrii=get_pk_r(field_real_coupled[:,:,i],field_real_coupled[:,:,i])
            lrr,prrjj=get_pk_r(field_real_coupled[:,:,j],field_real_coupled[:,:,j])
            plt.plot(lrr,prrij/np.sqrt(prrii*prrjj))
            plt.plot(larr,ctot[:,i,j]/np.sqrt(ctot[:,i,i]*ctot[:,j,j]))
            plt.ylim([0,1.1])
        plt.show()

    plt.figure()
    for i in np.arange(nfields) :
        lrr,prrii=get_pk_r(field_real_decoupled[:,:,i],field_real_decoupled[:,:,i])
        plt.plot(lrr,prrii)
        plt.plot(larr,cll[:,i])
    plt.show()
        
    #Filter
    field_real_coupled_filt=np.zeros([ng,ng,nfields])
    field_real_decoupled_filt=np.zeros([ng,ng,nfields])
    for i in np.arange(nfields) :
        field_real_coupled_filt[:,:,i]=np.real(np.fft.fft2(filters[:,:,i]*
                                                           np.fft.ifft2(field_real_coupled[:,:,i])))
        sig=np.std(field_real_coupled_filt[:,:,i])
        plt.figure(); plt.imshow(field_real_coupled_filt[:,:,i],interpolation='nearest',origin='lower',
                                 vmin=-3*sig,vmax=3*sig)
    plt.show()
    for i in np.arange(nfields) :
        field_real_decoupled_filt[:,:,i]=np.real(np.fft.fft2(filters_dc[:,:,i]*
                                                            np.fft.ifft2(field_real_decoupled[:,:,i])))
        sig=np.std(field_real_decoupled_filt[:,:,i])
        plt.figure(); plt.imshow(field_real_decoupled_filt[:,:,i],interpolation='nearest',origin='lower',
                                 vmin=-3*sig,vmax=3*sig)

    plt.show()

gen_random_real(c_p_fid,e_o,256,c_ij_fid-n_ij_fid,c_ij_fid)

om_arr=OC0-DOM+2*DOM*(np.arange(NOM)+0.5)/NOM
print om_arr
for i in np.arange(NOM) :
    fname='cl_ccl_om%03d_w000_s8000'%i
    print fname
    if os.path.isfile(fname+'.npy') :
        c=np.load(fname+'.npy')
    else :
        om=om_arr[i]
        fb=OB0/(OB0+OC0)
        ob=fb*om
        oc=om-ob
        c=cgf.compute_cls(oc,ob,HH0,S80,NS0,par0,nbins,zarr,nz_bins,LMAX)
        np.save(fname,c)

exit(1)

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
    plt.xlim([2,2000])
    plt.ylim([0.5,300])
    plt.loglog()
    plt.savefig('../Draft/Figs/d_p_wl.pdf',bbox_inches='tight')


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
    for i in (1+np.arange(199))*10 :
        ax.plot(zbarr,e_o[  i,:,0]*np.sqrt(ndens)/np.sqrt(np.sum(e_o[  i,:,0]**2*ndens)),'o-',markeredgewidth=0,
                color=cm.winter((i+0.5)/2001))
        if e_o[i,2,1]>0 :
            sign=-1
        else :
            sign=1
        ax.plot(zbarr,sign*e_o[  i,:,1]*np.sqrt(ndens)/np.sqrt(np.sum(e_o[  i,:,1]**2*ndens)),'o-',
                markeredgewidth=0,color=cm.autumn((i+0.5)/2001))
    plt.xlabel('$z_\\alpha$',fontsize=18)
    plt.ylabel('$\\sqrt{\\bar{n}^\\alpha}\\,({\\sf E}_\\ell)^1_\\alpha$',fontsize=18)
    plt.xlim([0.5,2.3])
    plt.savefig('../Draft/Figs/kl_modes_wl.pdf',bbox_inches='tight')

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
    plt.ylim([3E-7,1.2])
    plt.savefig('../Draft/Figs/information_wl.pdf',bbox_inches='tight')
    
if plot_stuff :
    plt.show()
