import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import erf
import pyccl as ccl
from scipy.integrate import quad


SZ_RED=0.03
LMAX=500

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
data=np.loadtxt("nz_red.txt",unpack=True)
nz_red=interp1d(data[0],data[1],bounds_error=False,fill_value=0)
zarr=np.linspace(0,2,1024)
bzarr=bz_red(zarr)
nzarr=nz_red(zarr)

#Selection function for individual bins
z0bins,zfbins=get_edges(0.1,1.4,SZ_RED,1.)
nbins=len(z0bins)
nz_bins=np.array([nzarr*pdf_photo(zarr,z0,zf,sz_red(0.5*(z0+zf))) for z0,zf in zip(z0bins,zfbins)])
xcorr=np.array([[np.sum(nz1*nz2) for nz1 in nz_bins] for nz2 in nz_bins])
xcorr/=np.sqrt(np.diag(xcorr)[:,None]*np.diag(xcorr)[None,:])

ndens=np.array([np.sum(nz)*(zarr[1]-zarr[0]) for nz in nz_bins])
print quad(nz_red,0,5)[0],np.sum(nzarr*(zarr[1]-zarr[0])),np.sum(ndens)
plt.figure(); plt.title("nz")
for nz in nz_bins :
    plt.plot(zarr,nz)
plt.plot(zarr,nzarr)

def compute_cls(oc,ob,h,s8,ns,w,fname_out=False) :
    #Fiducial cosmological parameters
    cosmo=ccl.Cosmology(Omega_c=oc,Omega_b=ob,h=h,sigma8=s8,n_s=ns,w0=w,
                        transfer_function='eisenstein_hu')
    print ccl.sigma8(cosmo)

    #Tracers
    tracers=[]
    for i in np.arange(nbins) :
        print i
        tracers.append(ccl.ClTracer(cosmo,tracer_type='nc',z=zarr,n=nz_bins[i],bias=bzarr))

    #Power spectra
    c_ij=np.zeros([LMAX+1,nbins,nbins])
    for i1 in np.arange(nbins) :
        for i2 in np.arange(i1,nbins) :
            print i1,i2
            if xcorr[i1,i2]<1E-6 :
                c_ij[:,i1,i2]=0
            else :
                c_ij[:,i1,i2]=ccl.angular_cl(cosmo,tracers[i1],tracers[i2],np.arange(LMAX+1))#,l_limber=100)
            if i1!=i2 :
                c_ij[:,i2,i1]=c_ij[:,i1,i2]
    if fname_out!=False :
        np.save(fname_out,c_ij)
    return c_ij

c_ij_fid=compute_cls(0.27,0.045,0.69,0.83,0.96,-1.00,'cl_fid')
c_ij_mw0=compute_cls(0.27,0.045,0.69,0.83,0.96,-1.05,'cl_mw0')
c_ij_pw0=compute_cls(0.27,0.045,0.69,0.83,0.96,-0.95,'cl_pw0')

c_ij_fid=np.load('cl_fid.npy')
c_ij_mw0=np.load('cl_mw0.npy')
c_ij_pw0=np.load('cl_pw0.npy')
n_ij_fid=np.zeros_like(c_ij_fid)
for i1 in np.arange(nbins) :
    n_ij_fid[:,i1,i1]=(np.pi/180./60.)**2/ndens[i1]
    c_ij_fid[:,i1,i1]+=n_ij_fid[:,i1,i1]
inv_cij=np.linalg.inv(c_ij_fid)
metric=np.linalg.inv(n_ij_fid)
c_ij_dw0=(c_ij_pw0-c_ij_mw0)/0.1
larr=np.arange(LMAX+1)
dprod=np.array([np.dot(c_ij_dw0[l,:,:],inv_cij[l,:,:]) for l in np.arange(LMAX+1)])
fish=(larr+0.5)*np.array([np.trace(np.dot(d,d)) for d in dprod])
sigma=np.sqrt(1./np.sum(fish))

plt.figure(); plt.title("cij")
cols=['r','g','b','y','m','c']
for i1 in np.arange(nbins) :
    plt.plot(larr,c_ij_fid[:,i1,i1],cols[i1%6]+'-' )
plt.loglog()

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

e_v,c_p_fid=diagonalize(c_ij_fid,np.linalg.inv(n_ij_fid))
plt.figure(); plt.title("first")
for c in np.transpose(c_p_fid) :
    plt.plot(larr,c)
plt.loglog()

e_o=e_v
#e_o=np.zeros_like(e_v)
#e_o[0,:,:]=e_v[0,:,:]
#for l in np.arange(LMAX)+1 :
#    i_all=np.arange(nbins)
#    indices=np.arange(nbins)
##    if l>150 : indices=np.argsort(-c_p_fid[l])
##    if l==211 :
##        print c_p_fid[l,indices]
##        print c_p_fid[l-1,indices]
##        exit(1)
#    for a in np.arange(nbins) :
#        id=np.argmin(np.fabs(c_p_fid[l-1,indices[a]]-c_p_fid[l,indices[i_all]]))
#        e_o[l,:,indices[a]]=e_v[l,:,i_all[id]]
#        i_all=np.delete(i_all,id)
#for l in np.arange(LMAX)+1 :
#    i_all=np.arange(nbins)
#    indices=np.argsort(-c_p_fid[l])
#    if l==5 :
#        print c_p_fid[l-1], c_p_fid[l]
#    for a in np.arange(nbins) :
#        id=np.argmin(np.fabs(c_p_fid[l-1,indices[a]]-c_p_fid[l,i_all]))
#        if l==5 :
#            print a,indices[a],i_all[id],id,c_p_fid[l-1,indices[a]],c_p_fid[l,i_all[id]]
#        e_o[l,:,a]=e_v[l,:,i_all[id]]
#        i_all=np.delete(i_all,id)
##        exit(1)

c_p_fid_o=change_basis(c_ij_fid,metric,e_o)
c_p_dw0  =change_basis(c_ij_dw0,metric,e_o)
fisher=(larr+0.5)[:,None]*(c_p_dw0/c_p_fid_o)**2

plt.figure(); plt.title("reorder")
for c in np.transpose(c_p_fid_o) :
    plt.plot(larr,c)
plt.loglog()

isort=np.argsort(-np.sum(fisher,axis=0))
e_o=e_o[:,:,isort]

c_p_fid_o=change_basis(c_ij_fid,metric,e_o)
c_p_dw0  =change_basis(c_ij_dw0,metric,e_o)
n_p_fid  =change_basis(n_ij_fid,metric,e_o)
fisher=(larr+0.5)[:,None]*(c_p_dw0/c_p_fid_o)**2

plt.figure(); plt.title("reorder fisher")
for c in np.transpose(c_p_fid_o) :
    plt.plot(larr,c)
plt.loglog()

fish_permode=np.sum(fisher,axis=0)
fish_cum=np.cumsum(fish_permode)

#plt.figure()
#for c in np.transpose(n_p_fid) :
#    plt.plot(larr,c)
#plt.loglog()

plt.figure(); plt.plot(fish_cum/fish_cum[-1]); plt.plot(fish_permode/np.amax(fish_permode))

plt.show()
