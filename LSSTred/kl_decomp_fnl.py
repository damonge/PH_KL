import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import erf
import pyccl as ccl
from scipy.integrate import quad
import matplotlib.cm as cm
import common_gofish as cgf
import os

plot_stuff=True
SZ_RED=0.02
LMAX=500
nsamp=0.5
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
zbarr=0.5*(z0bins+zfbins)
nbins=len(z0bins)
nz_bins=np.array([nzarr*pdf_photo(zarr,z0,zf,sz_red(0.5*(z0+zf))) for z0,zf in zip(z0bins,zfbins)])
xcorr=np.array([[np.sum(nz1*nz2) for nz1 in nz_bins] for nz2 in nz_bins])
xcorr/=np.sqrt(np.diag(xcorr)[:,None]*np.diag(xcorr)[None,:])

#Compute number densities
ndens=np.array([np.sum(nz)*(zarr[1]-zarr[0]) for nz in nz_bins])
print quad(nz_red,0,5)[0],np.sum(nzarr*(zarr[1]-zarr[0])),np.sum(ndens)

run_name="sz%.2lf_"%SZ_RED+"ns%d_"%(int(nsamp))+"lmx%d"%LMAX
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
    plt.ylim([0,1.05*np.amax(nzarr)])
    plt.xlim([0,1.2*zedge_hi])
    plt.xlabel('$z$',fontsize=18)
    plt.ylabel('$N(z)\\,\\,[{\\rm arcmin}^{-2}]$',fontsize=18)

#Compute power spectra
c_ij_fid,c_ij_mfn,c_ij_pfn=cgf.run_gofish(run_name,LMAX,parname,par0,dpar,tracertype,marg_all=False);
n_ij_fid=np.zeros_like(c_ij_fid)
for i1 in np.arange(nbins) :
    n_ij_fid[:,i1,i1]=sigma_gamma**2*(np.pi/180./60.)**2/ndens[i1]
    c_ij_fid[:,i1,i1]+=n_ij_fid[:,i1,i1]
c_ij_dfn=(c_ij_pfn-c_ij_mfn)/(2*dpar)
larr=np.arange(LMAX+1)
inv_cij=np.linalg.inv(c_ij_fid)
in_ij_fid=np.linalg.inv(n_ij_fid)
metric=in_ij_fid
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
    plt.xlim([2,500])
    plt.ylim([5E-7,1E-4])
    plt.savefig('../Draft/Figs/c_ij_gc.pdf',bbox_inches='tight')

def diagonalize(dc,c) :
    mm=np.linalg.cholesky(np.linalg.inv(c))
    dcp=np.array([np.dot(np.transpose(mm[l]),np.dot(dc[l],mm[l])) for l in np.arange(LMAX+1)])
    dc_p,e=np.linalg.eigh(-dcp); dc_p*=-1
    v=np.array([np.dot(mm[l],e[l]) for l in np.arange(LMAX+1)])
    
    return v,dc_p

#Get K-L modes
vv2,llam2=diagonalize(c_ij_dfn,c_ij_fid)
isort=np.argsort(-np.sum((larr+0.5)[:,None]*llam2**2,axis=0))
vv=vv2[:,:,isort]
llam=np.array([np.diag(np.dot(np.transpose(vv[l]),np.dot(c_ij_dfn[l],vv[l]))) for l in np.arange(LMAX+1)])
fishers=np.sum((larr+0.5)[:,None]*llam**2,axis=0)
sigma_cumul=np.sqrt(1./np.cumsum(fishers))

def get_sigma_tomo(ntomo) :
    dn=nbins/ntomo
    fs=[]
    n0=0
    while n0<nbins :
        if n0+2*dn<=nbins :
            nmax=n0+dn
        else :
            nmax=nbins
        msk=np.zeros(nbins); msk[n0:nmax]=1; f=ndens*msk/np.sqrt(np.sum((ndens*msk)**2))
        fs.append(f)
        n0=nmax
    f_tm=np.transpose(np.array(fs))[None,:,:]*(np.ones(LMAX+1))[:,None,None]
    fish=cgf.get_fisher_dd(run_name,f_tm,n_ij_fid,do_print=False)
    return 1./np.sqrt(np.sum(fish))

sigma_tomo=np.array([get_sigma_tomo(i) for i in np.arange(nbins)+1])
plt.figure(); plt.plot(fishers/np.sum(fishers)); plt.plot(1-np.cumsum(fishers)/np.sum(fishers))
plt.figure(); plt.plot(sigma_cumul/sigma_cumul[-1]); plt.plot(sigma_tomo/sigma_cumul[-1])
#np.savetxt("sigmas_kl.txt",np.transpose([np.arange(nbins)+1,sigma_cumul]))
np.savetxt("sigmas_tm.txt",np.transpose([np.arange(nbins)+1,sigma_tomo]))
print sigma_tomo
plt.figure();
nplot=10
for i in np.arange(nplot) :
    f=(i+0.5)/nplot
    plt.plot(zbarr,vv[10,:,i],'-',lw=2,color=cm.bone(f))
plt.show()

'''
e_v,c_p_fid=diagonalize(c_ij_fid,np.linalg.inv(n_ij_fid))
c_p_dfn  =change_basis(c_ij_dfn,metric,e_v)
fisher=(larr+0.5)[:,None]*(c_p_dfn/c_p_fid)**2
isort=np.argsort(-np.sum(fisher,axis=0))
e_o=e_v[:,:,isort]
f_o=np.array([np.dot(in_ij_fid[l],e_o[l,:,:]) for l in np.arange(LMAX+1)])
c_p_fid=change_basis(c_ij_fid,metric,e_o)
c_p_dfn=change_basis(c_ij_dfn,metric,e_o)

#Plot power spectrum of K-L modes
if plot_stuff :
    plt.figure();
    ax=plt.gca()
    ax.imshow([[0.,1.],[0.,1.]],extent=[230,430,270,380],interpolation='bicubic',cmap=cm.summer,aspect='auto')
    plt.text(90,300,'$p\\in[1,%d]$'%nbins,{'fontsize':16})
    for i in np.arange(nbins) :
        c=c_p_fid[:,i]
        col=cm.summer((i+0.5)/nbins)
        plt.plot(larr,c,color=col,lw=2)
    plt.ylabel("$D_\\ell^p$",fontsize=16)
    plt.xlabel("$\\ell$",fontsize=16)
    plt.xlim([2,500])
    plt.ylim([0.8,500])
    plt.loglog()
    plt.savefig('../Draft/Figs/d_p_gc.pdf',bbox_inches='tight')
'''

f_o=vv
#Tomography, 1 bin
msk1=np.zeros(nbins); msk1[:]=1; f1=ndens*msk1/np.sqrt(np.sum((ndens*msk1)**2));
f_tm1=f1[None,:,None]*np.ones([LMAX+1,nbins,1])
#Tomograpy, 2 bins
msk1=np.zeros(nbins); msk1[:12]=1; f1=ndens*msk1/np.sqrt(np.sum((ndens*msk1)**2));
msk2=np.zeros(nbins); msk2[12:]=1; f2=ndens*msk2/np.sqrt(np.sum((ndens*msk2)**2));
f_tm2=(np.transpose([f1,f2]))[None,:,:]*(np.ones(LMAX+1))[:,None,None];
#Tomograpy, 3 bins
msk1=np.zeros(nbins); msk1[:8]  =1; f1=ndens*msk1/np.sqrt(np.sum((ndens*msk1)**2));
msk2=np.zeros(nbins); msk2[8:16]=1; f2=ndens*msk2/np.sqrt(np.sum((ndens*msk2)**2));
msk3=np.zeros(nbins); msk3[16:] =1; f3=ndens*msk3/np.sqrt(np.sum((ndens*msk3)**2));
f_tm3=(np.transpose([f1,f2,f3]))[None,:,:]*(np.ones(LMAX+1))[:,None,None];
#Tomograpy, 4 bins
msk1=np.zeros(nbins); msk1[:6]  =1; f1=ndens*msk1/np.sqrt(np.sum((ndens*msk1)**2));
msk2=np.zeros(nbins); msk2[6:12] =1; f2=ndens*msk2/np.sqrt(np.sum((ndens*msk2)**2));
msk3=np.zeros(nbins); msk3[12:18]=1; f3=ndens*msk3/np.sqrt(np.sum((ndens*msk3)**2));
msk4=np.zeros(nbins); msk4[18:] =1; f4=ndens*msk4/np.sqrt(np.sum((ndens*msk4)**2));
f_tm4=(np.transpose([f1,f2,f3,f4]))[None,:,:]*(np.ones(LMAX+1))[:,None,None];
#Tomograpy, 5 bins
msk1=np.zeros(nbins); msk1[:5]   =1; f1=ndens*msk1/np.sqrt(np.sum((ndens*msk1)**2));
msk2=np.zeros(nbins); msk2[5:9]  =1; f2=ndens*msk2/np.sqrt(np.sum((ndens*msk2)**2));
msk3=np.zeros(nbins); msk3[9:14] =1; f3=ndens*msk3/np.sqrt(np.sum((ndens*msk3)**2));
msk4=np.zeros(nbins); msk4[14:19]=1; f4=ndens*msk4/np.sqrt(np.sum((ndens*msk4)**2));
msk5=np.zeros(nbins); msk5[19:]  =1; f5=ndens*msk5/np.sqrt(np.sum((ndens*msk5)**2));
f_tm5=(np.transpose([f1,f2,f3,f4,f5]))[None,:,:]*(np.ones(LMAX+1))[:,None,None];
#Tomograpy, 6 bins
msk1=np.zeros(nbins); msk1[:4]   =1; f1=ndens*msk1/np.sqrt(np.sum((ndens*msk1)**2));
msk2=np.zeros(nbins); msk2[4:8]  =1; f2=ndens*msk2/np.sqrt(np.sum((ndens*msk2)**2));
msk3=np.zeros(nbins); msk3[8:12] =1; f3=ndens*msk3/np.sqrt(np.sum((ndens*msk3)**2));
msk4=np.zeros(nbins); msk4[12:16]=1; f4=ndens*msk4/np.sqrt(np.sum((ndens*msk4)**2));
msk5=np.zeros(nbins); msk5[16:20]=1; f5=ndens*msk5/np.sqrt(np.sum((ndens*msk5)**2));
msk6=np.zeros(nbins); msk6[20:]  =1; f6=ndens*msk6/np.sqrt(np.sum((ndens*msk6)**2));
f_tm6=(np.transpose([f1,f2,f3,f4,f5,f6]))[None,:,:]*(np.ones(LMAX+1))[:,None,None];
#Full tomography
f_id=((np.identity(nbins))[None,:,:])*((np.ones(LMAX+1))[:,None,None])
print "KL, full",
fish_kl_full=cgf.get_fisher_dd(run_name,f_o,n_ij_fid)
print "TM, Full",
fish_tm_full=cgf.get_fisher_dd(run_name,f_id,n_ij_fid)
print "KL, 1",
fish_kl_1=cgf.get_fisher_dd(run_name,f_o[:,:,0:1],n_ij_fid)
print "TM, 1",
fish_tm_1=cgf.get_fisher_dd(run_name,f_tm1,n_ij_fid)
print "KL, 2",
fish_kl_2=cgf.get_fisher_dd(run_name,f_o[:,:,0:2],n_ij_fid)
print "TM, 2",
fish_tm_2=cgf.get_fisher_dd(run_name,f_tm2,n_ij_fid)
print "KL, 3",
fish_kl_3=cgf.get_fisher_dd(run_name,f_o[:,:,0:3],n_ij_fid)
print "TM, 3",
fish_tm_3=cgf.get_fisher_dd(run_name,f_tm3,n_ij_fid)
print "KL, 4",
fish_kl_4=cgf.get_fisher_dd(run_name,f_o[:,:,0:4],n_ij_fid)
print "TM, 4",
fish_tm_4=cgf.get_fisher_dd(run_name,f_tm4,n_ij_fid)
print "KL, 5",
fish_kl_5=cgf.get_fisher_dd(run_name,f_o[:,:,0:5],n_ij_fid)
print "TM, 5",
fish_tm_5=cgf.get_fisher_dd(run_name,f_tm5,n_ij_fid)
print "KL, 6",
fish_kl_6=cgf.get_fisher_dd(run_name,f_o[:,:,0:6],n_ij_fid)
print "TM, 6",
fish_tm_6=cgf.get_fisher_dd(run_name,f_tm6,n_ij_fid)

exit(1)

#Plot K-L eigenvectors
if plot_stuff :
    plt.figure()
    ax=plt.gca()
    ax.imshow([[0.,1.],[0.,1.]],extent=[1.25,1.37,-0.22,-0.17],interpolation='bicubic',
              cmap=cm.winter,aspect='auto')
    nbtop=7
    i_ell=30
    plt.text(1.12,-0.203,'$p\\in[1,%d]$'%nbtop,{'fontsize':16})
    for i in np.arange(nbtop) :
        ax.plot(zbarr,e_o[i_ell,:,i]*np.sqrt(ndens)/np.sqrt(np.sum(e_o[i_ell,:,i]**2*ndens)),'o-',
                markeredgewidth=0,color=cm.winter((i+0.5)/nbtop))
    plt.xlabel('$z_\\alpha$',fontsize=18)
    plt.ylabel('$\\sqrt{\\bar{n}^\\alpha}\\,({\\sf F}_{%d})^p_\\alpha$'%i_ell,fontsize=18)
    plt.savefig('../Draft/Figs/kl_modes_gc.pdf',bbox_inches='tight')


fisher=(larr+0.5)[:,None]*(c_p_dfn/c_p_fid)**2
fish_permode=np.sum(fisher,axis=0)
fish_cum=np.cumsum(fish_permode)

if plot_stuff :
    plt.figure();
    imodes=np.arange(nbins)+1
    plt.plot(imodes,fish_permode/np.sum(fish_permode),'go-',lw=2,
             label='${\\rm Information\\,\\,in\\,\\,mode}\\,\\,p_{\\rm KL}$',markeredgewidth=0)
    plt.plot(imodes[:-1],1-fish_cum[:-1]/fish_cum[-1],'ro-',lw=2,label='${\\rm Information\\,\\,in\\,\\,modes}\\,\\,>p_{\\rm KL}$',markeredgewidth=0)
    plt.legend(loc='upper right',frameon=False)
    plt.xlabel('${\\rm KL\\,\\,mode\\,\\,order}\\,\\,p_{\\rm KL}$',fontsize=18)
    plt.ylabel('${\\rm Relative\\,information\\,\\,content}$',fontsize=18)
    plt.xlim([0.9,nbins+0.1])
    plt.ylim([0,1.0])
    plt.savefig('../Draft/Figs/information_gc.pdf',bbox_inches='tight')

if plot_stuff :
    plt.show()
