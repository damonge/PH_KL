import numpy as np
import os
import struct
import array
import pyccl as ccl

def read_cls_class(fname) :
    f=open(fname,"rd")
    data=f.read()
    f.close()

    lmax,n_nc,n_wl=struct.unpack('3I',data[0:12])
    has_tt,has_ee,has_te,has_bb,has_pp,has_tp,has_ep=struct.unpack('7I',data[12:12+7*4])
    has_td,has_tl,has_pd,has_pl,has_dd,has_dl,has_ll=struct.unpack('7I',data[12+7*4:12+14*4])
    offset_header=17*4
    n_ct=0
    if has_tt :
        n_ct+=1
    if has_ee :
        n_ct+=1
    if has_te :
        n_ct+=1
    if has_bb :
        n_ct+=1
    if has_pp :
        n_ct+=1
    if has_tp :
        n_ct+=1
    if has_ep :
        n_ct+=1
    if has_td :
        n_ct+=n_nc
    if has_tl :
        n_ct+=n_wl
    if has_pd :
        n_ct+=n_nc
    if has_pl :
        n_ct+=n_wl
    if has_dd :
        n_ct+=n_nc*(n_nc+1)/2
    if has_dl :
        n_ct+=n_nc*n_wl
    if has_ll :
        n_ct+=n_wl*(n_wl+1)/2
    len_cls=8*n_ct*(lmax-1)

    if len(data)!=len_cls+offset_header :
        strout="Error reading CLASS file "+fname
        strout+="%.2lf"%((len(data)-offset_header)/8.)+" %.2lf"%(len_cls/8.)
        sys.exit(strout)

    dum=array.array('d')
    dum.fromstring(data[offset_header:])#.
    cldata=np.array(dum).reshape((lmax-1,n_ct))

    cl_tt=None; cl_ee=None; cl_te=None; cl_bb=None;
    cl_pp=None; cl_tp=None; cl_ep=None; 
    cl_td=None; cl_tl=None; cl_pd=None; cl_pl=None;
    cl_dd=None; cl_dl=None; cl_ll=None; 

    offset_here=0
    larr=np.arange(lmax-1)+2
    prefac=2*np.pi/(larr*(larr+1))
    prefac_cmbl=0.5*larr*(larr+1)
    if has_tt :
        cl_tt=np.zeros(lmax+1)
        cl_tt[2:]=prefac*cldata[:,offset_here]
        offset_here+=1
    if has_ee :
        cl_ee=np.zeros(lmax+1)
        cl_ee[2:]=prefac*cldata[:,offset_here]
        offset_here+=1
    if has_te :
        cl_te=np.zeros(lmax+1)
        cl_te[2:]=prefac*cldata[:,offset_here]
        offset_here+=1
    if has_bb :
        cl_bb=np.zeros(lmax+1)
        cl_bb[2:]=prefac*cldata[:,offset_here]
        offset_here+=1
    if has_pp :
        cl_pp=np.zeros(lmax+1)
        cl_pp[2:]=prefac*prefac_cmbl**2*cldata[:,offset_here]
        offset_here+=1
    if has_tp :
        cl_tp=np.zeros(lmax+1)
        cl_tp[2:]=prefac*prefac_cmbl*cldata[:,offset_here]
        offset_here+=1
    if has_ep :
        cl_ep=np.zeros(lmax+1)
        cl_ep[2:]=prefac*prefac_cmbl*cldata[:,offset_here]
        offset_here+=1
    if has_td :
        cl_td=np.zeros([lmax+1,n_nc])
        for i in np.arange(n_nc) :
            cl_td[2:,i]=prefac*cldata[:,offset_here+i]
        offset_here+=n_nc
    if has_tl :
        cl_tl=np.zeros([lmax+1,n_wl]) 
        for i in np.arange(n_wl) :
            cl_tl[2:,i]=prefac*cldata[:,offset_here+i]
        offset_here+=n_wl
    if has_pd :
        cl_pd=np.zeros([lmax+1,n_nc])
        for i in np.arange(n_nc) :
            cl_pd[2:,i]=prefac*prefac_cmbl*cldata[:,offset_here+i]
        offset_here+=n_nc
    if has_pl :
        cl_pl=np.zeros([lmax+1,n_wl])
        for i in np.arange(n_wl) :
            cl_pl[2:,i]=prefac*prefac_cmbl*cldata[:,offset_here+i]
        offset_here+=n_wl
    if has_dd :
        cl_dd=np.zeros([lmax+1,n_nc,n_nc])
        for i in np.arange(n_nc) :
            for j in np.arange(n_nc-i)+i :
                cl_dd[2:,i,j]=prefac*cldata[:,offset_here]
                if i!=j :
                    cl_dd[2:,j,i]=prefac*cldata[:,offset_here]
                offset_here+=1
    if has_dl :
        cl_dl=np.zeros([lmax+1,n_nc,n_wl])
        for i in np.arange(n_nc) :
            for j in np.arange(n_wl) :
                cl_dl[2:,i,j]=prefac*cldata[:,offset_here]
                offset_here+=1
    if has_ll :
        cl_ll=np.zeros([lmax+1,n_wl,n_wl])
        for i in np.arange(n_wl) :
            for j in np.arange(n_wl-i)+i :
                cl_ll[2:,i,j]=prefac*cldata[:,offset_here]
                if i!=j :
                    cl_ll[2:,j,i]=prefac*cldata[:,offset_here]
                offset_here+=1

    #Wrap into a dictionary
    dic={'lmax':lmax,'n_nc':n_nc,'n_wl':n_wl}
    dic.update({'has_tt':has_tt,'cl_tt':cl_tt})
    dic.update({'has_ee':has_ee,'cl_ee':cl_ee})
    dic.update({'has_te':has_te,'cl_te':cl_te})
    dic.update({'has_bb':has_bb,'cl_bb':cl_bb})
    dic.update({'has_pp':has_pp,'cl_pp':cl_pp})
    dic.update({'has_tp':has_tp,'cl_tp':cl_tp})
    dic.update({'has_ep':has_ep,'cl_ep':cl_ep})
    dic.update({'has_td':has_td,'cl_td':cl_td})
    dic.update({'has_tl':has_tl,'cl_tl':cl_tl})
    dic.update({'has_pd':has_pd,'cl_pd':cl_pd})
    dic.update({'has_pl':has_pl,'cl_pl':cl_pl})
    dic.update({'has_dd':has_dd,'cl_dd':cl_dd})
    dic.update({'has_dl':has_dl,'cl_dl':cl_dl})
    dic.update({'has_ll':has_ll,'cl_ll':cl_ll})

    return dic

def run_gofish(rname,lmx,parname,par0,dpar,trtype,w_IA=False,llim=None) :
    if llim==None :
        if trtype=='gal_clustering' :
            l_limber=lmx+1
        else :
            l_limber=100
    else :
        l_limber=llim

    print "WOO"
    stout=""
#    stout+='[och2]\n'
#    stout+='x= 0.1197\n'
#    stout+='dx=0.0010\n'
#    stout+='is_free=yes\n'
#    stout+='onesided=0\n'
#    stout+='\n'
#    stout+='[A_s]\n'
#    stout+='x= 2.19\n'
#    stout+='dx=0.01\n'
#    stout+='is_free=yes\n'
#    stout+='onesided=0\n'
#    stout+='\n'
    stout+='['+parname+']\n'
    stout+='x= %lE\n'%par0
    stout+='dx=%lE\n'%dpar
    stout+='is_free=yes\n'
    stout+='onesided=0\n'
    stout+='\n'
    stout+='[Tracer 1]\n'
    stout+='tracer_name=gal_survey \n'
    stout+='tracer_type='+trtype+' \n'
    stout+='bins_file=../bins_'+rname+'.txt\n'
    stout+='nz_file=../nz_'+rname+'.txt\n'
    if trtype=='gal_clustering' :
        stout+='bias_file=../bz_'+rname+'.txt\n'
        stout+='sbias_file= stupid\n'
        stout+='ebias_file= stupid\n'
    else :
        stout+='abias_file= curves_LSST/az_gold.txt\n'
        stout+='rfrac_file= curves_LSST/rf_gold.txt\n'
        stout+='sigma_gamma= 0.28\n'
    stout+='use_tracer= yes\n'
    stout+='\n'
    stout+='[CLASS parameters]\n'
    stout+='lmax_lss= %d\n'%lmx
    stout+='lmin_limber= %d\n'%l_limber
    if w_IA :
        stout+='include_alignment= yes\n'
    else :
        stout+='include_alignment= no\n'
    stout+='include_rsd= no\n'
    stout+='include_magnification= no\n'
    stout+='include_gr_vel= no\n'
    stout+='include_gr_pot= no\n'
#    stout+='exec_path= addqueue -q cmb -s -n 1x12 -m 2 -c "gofish" ./class_mod\n'
    stout+='exec_path= ./class_mod\n'
    stout+='use_nonlinear= yes\n'
    stout+='f_sky= 1.0\n'
    stout+='\n'
    stout+='[Output parameters]\n'
    stout+='output_dir= ../outputs_'+rname+'\n'
    stout+='output_spectra= run\n'
    stout+='output_fisher= Fisher\n'
    stout+='\n'
    stout+='[Behaviour parameters]\n'
    stout+='model= wCDM\n'
    stout+='save_cl_files= yes\n'
    stout+='save_param_files= yes\n'
    f=open('param_'+rname+'.ini','w')
    f.write(stout)
    f.close()
    os.system('cd GoFish ; python main.py ../param_'+rname+'.ini; cd ..')

    dic_fid=read_cls_class('outputs_'+rname+'/run_fidcl.dat')
    dic_mfn=read_cls_class('outputs_'+rname+'/run_m'+parname+'cl.dat')
    dic_pfn=read_cls_class('outputs_'+rname+'/run_p'+parname+'cl.dat')

    if trtype=='gal_clustering' :
        return dic_fid['cl_dd'],dic_mfn['cl_dd'],dic_pfn['cl_dd'],
    else :
        return dic_fid['cl_ll'],dic_mfn['cl_ll'],dic_pfn['cl_ll'],


#def compute_cls(oc,ob,h,a_s,ns,w,nbins,zarr,nz_bins,lmax,fname_out=False) :
#    cosmo=ccl.Cosmology(Omega_c=oc,Omega_b=ob,h=h,A_s=a_s,n_s=ns,w0=w)#,transfer_function='eisenstein_hu')
def compute_cls(oc,ob,h,s8,ns,w,nbins,zarr,nz_bins,lmax,fname_out=False) :
    cosmo=ccl.Cosmology(Omega_c=oc,Omega_b=ob,h=h,sigma8=s8,n_s=ns,w0=w)#,transfer_function='eisenstein_hu')
    print ccl.sigma8(cosmo)

    #Tracers
    tracers=[]
    for i in np.arange(nbins) :
        tracers.append(ccl.ClTracer(cosmo,tracer_type='wl',z=zarr,n=nz_bins[i]))

    #Power spectra
    c_ij=np.zeros([lmax+1,nbins,nbins])
    for i1 in np.arange(nbins) :
        print i1
        for i2 in np.arange(i1,nbins) :
            c_ij[:,i1,i2]=ccl.angular_cl(cosmo,tracers[i1],tracers[i2],np.arange(lmax+1))#,l_limber=100)
            if i1!=i2 :
                c_ij[:,i2,i1]=c_ij[:,i1,i2]
    if fname_out!=False :
        np.save(fname_out,c_ij)
    return c_ij
