import numpy as np
import os
import struct
import array
import pyccl as ccl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import ScalarFormatter

FS=18

formatter=ScalarFormatter(useOffset=False)

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

def run_gofish(rname,lmx,parname,par0,dpar,trtype,w_IA=False,llim=None,marg_all=True) :
    if llim==None :
        if trtype=='gal_clustering' :
            l_limber=lmx+1
        else :
            l_limber=100
    else :
        l_limber=llim

    print "WOO"
    stout=""
    if marg_all :
        stout+='[och2]\n'
        stout+='x= 0.1197\n'
        stout+='dx=0.0010\n'
        stout+='is_free=yes\n'
        stout+='onesided=0\n'
        stout+='\n'
        stout+='[obh2]\n'
        stout+='x= 0.02222\n'
        stout+='dx=0.0001\n'
        stout+='is_free=yes\n'
        stout+='onesided=0\n'
        stout+='\n'
        stout+='[hh]\n'
        stout+='x= 0.67\n'
        stout+='dx=0.01\n'
        stout+='is_free=yes\n'
        stout+='onesided=0\n'
        stout+='\n'
        stout+='[ns]\n'
        stout+='x= 0.96\n'
        stout+='dx=0.01\n'
        stout+='is_free=yes\n'
        stout+='onesided=0\n'
        stout+='\n'
        stout+='[A_s]\n'
        stout+='x= 2.19\n'
        stout+='dx=0.01\n'
        stout+='is_free=yes\n'
        stout+='onesided=0\n'
        stout+='\n'
        stout+='[mnu]\n'
        stout+='x= 60.0\n'
        stout+='dx= 5.0\n'
        stout+='is_free=yes\n'
        stout+='onesided=0\n'
        stout+='\n'
        stout+='[lmcb]\n'
        stout+='x= 14.08\n'
        stout+='dx=0.3\n'
        stout+='is_free=yes\n'
        stout+='onesided=0\n'
        stout+='\n'
        stout+='[etab]\n'
        stout+='x= 0.5\n'
        stout+='dx=0.1\n'
        stout+='is_free=yes\n'
        stout+='onesided=0\n'
        stout+='\n'
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
    stout+='use_baryons= yes\n'
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

def get_fisher_ll(rname,mat_project,nij,marg_all=True) :
    nl=len(mat_project)
    nel=len(mat_project[0,0])
    if marg_all :
        npar=9
        fisher_prior=np.diag(1./(np.array([1E6,1.5E-3,1.0E-2,1.6E-4,5E-3,1E-2, 1E6, 1E6,1E6]))**2)
#        fisher_prior=np.diag(1./(np.array([1E6,  1E-6,  1E-6,  1E-6,1E-6,1E-6,1E-6,1E-6,1E6]))**2)
#        fisher_prior=np.diag(1./(np.array([1E6,   1E6,   1E6,   1E6, 1E6, 1E6, 1E6, 1E6,1E6]))**2)
    else :
        npar=1
        fisher_prior=np.diag(1./(np.array([1E6]))**2)
    larr=np.arange(nl)
    def transform_matrix(c,m) :
        return np.array([np.dot(np.transpose(m[l]),np.dot(c[l],m[l])) for l in np.arange(nl)])
    dic=read_cls_class('outputs_'+rname+'/run_fidcl.dat');
    cl_ij_fid=transform_matrix(dic['cl_ll']+nij,mat_project)
    il_ij_fid=np.linalg.inv(cl_ij_fid)
    dl_all=np.zeros([npar,nl,nel,nel])
    ipar=0
    if marg_all :
    #mnu
        dic=read_cls_class('outputs_'+rname+'/run_mmnucl.dat'); cl_ij_mmnu=dic['cl_ll']
        dic=read_cls_class('outputs_'+rname+'/run_pmnucl.dat'); cl_ij_pmnu=dic['cl_ll']
        dl_all[ipar,:,:,:]=transform_matrix((cl_ij_pmnu-cl_ij_mmnu)/(2*5.),mat_project); ipar+=1
    #och2
        dic=read_cls_class('outputs_'+rname+'/run_moch2cl.dat'); cl_ij_moch2=dic['cl_ll']
        dic=read_cls_class('outputs_'+rname+'/run_poch2cl.dat'); cl_ij_poch2=dic['cl_ll']
        dl_all[ipar,:,:,:]=transform_matrix((cl_ij_poch2-cl_ij_moch2)/(2*0.001),mat_project); ipar+=1
    #hh
        dic=read_cls_class('outputs_'+rname+'/run_mhhcl.dat'); cl_ij_mhh=dic['cl_ll']
        dic=read_cls_class('outputs_'+rname+'/run_phhcl.dat'); cl_ij_phh=dic['cl_ll']
        dl_all[ipar,:,:,:]=transform_matrix((cl_ij_phh-cl_ij_mhh)/(2*0.01),mat_project); ipar+=1
    #obh2
        dic=read_cls_class('outputs_'+rname+'/run_mobh2cl.dat'); cl_ij_mobh2=dic['cl_ll']
        dic=read_cls_class('outputs_'+rname+'/run_pobh2cl.dat'); cl_ij_pobh2=dic['cl_ll']
        dl_all[ipar,:,:,:]=transform_matrix((cl_ij_pobh2-cl_ij_mobh2)/(2*0.0001),mat_project); ipar+=1
    #ns
        dic=read_cls_class('outputs_'+rname+'/run_mnscl.dat'); cl_ij_mns=dic['cl_ll']
        dic=read_cls_class('outputs_'+rname+'/run_pnscl.dat'); cl_ij_pns=dic['cl_ll']
        dl_all[ipar,:,:,:]=transform_matrix((cl_ij_pns-cl_ij_mns)/(2*0.01),mat_project); ipar+=1
    #A_s
        dic=read_cls_class('outputs_'+rname+'/run_mA_scl.dat'); cl_ij_mA_s=dic['cl_ll']
        dic=read_cls_class('outputs_'+rname+'/run_pA_scl.dat'); cl_ij_pA_s=dic['cl_ll']
        dl_all[ipar,:,:,:]=transform_matrix((cl_ij_pA_s-cl_ij_mA_s)/(2*0.01),mat_project); ipar+=1
    #lmcb
        dic=read_cls_class('outputs_'+rname+'/run_mlmcbcl.dat'); cl_ij_mlmcb=dic['cl_ll']
        dic=read_cls_class('outputs_'+rname+'/run_plmcbcl.dat'); cl_ij_plmcb=dic['cl_ll']
        dl_all[ipar,:,:,:]=transform_matrix((cl_ij_plmcb-cl_ij_mlmcb)/(2*0.3),mat_project); ipar+=1
    #etab
        dic=read_cls_class('outputs_'+rname+'/run_metabcl.dat'); cl_ij_metab=dic['cl_ll']
        dic=read_cls_class('outputs_'+rname+'/run_petabcl.dat'); cl_ij_petab=dic['cl_ll']
        dl_all[ipar,:,:,:]=transform_matrix((cl_ij_petab-cl_ij_metab)/(2*0.1),mat_project); ipar+=1
    #w0
    dic=read_cls_class('outputs_'+rname+'/run_mw0cl.dat'); cl_ij_mw0=dic['cl_ll']
    dic=read_cls_class('outputs_'+rname+'/run_pw0cl.dat'); cl_ij_pw0=dic['cl_ll']
    dl_all[ipar,:,:,:]=transform_matrix((cl_ij_pw0-cl_ij_mw0)/(2*0.05),mat_project); ipar+=1

#    fisher_prior=np.diag(1./(np.array([1E6,1E-6,1E-6,1E-6,1E-6,1E-6,1E-6,1E-6,1E6]))**2)

    for i1 in np.arange(npar) :
        for l in larr :
            dl_all[i1,l,:,:]=np.dot(dl_all[i1,l],il_ij_fid[l])

    fisher=np.zeros([npar,npar])
    for i1 in np.arange(npar) :
        d1=dl_all[i1]
        for i2 in np.arange(npar) :
            d2=dl_all[i2]
            fisher[i1,i2]=np.sum((larr+0.5)*np.array([np.trace(np.dot(d1[l],d2[l])) for l in larr]))
    fisher+=fisher_prior
    stout=""
    for s in np.sqrt(np.diag(np.linalg.inv(fisher))) :
        stout+="%.3lE "%s
    print stout

    return fisher

def get_fisher_dd(rname,mat_project,nij,do_print=True) :
    nl=len(mat_project)
    nel=len(mat_project[0,0])
    npar=1
    larr=np.arange(nl)
    def transform_matrix(c,m) :
        return np.array([np.dot(np.transpose(m[l]),np.dot(c[l],m[l])) for l in np.arange(nl)])
    dic=read_cls_class('outputs_'+rname+'/run_fidcl.dat');
    cl_ij_fid=transform_matrix(dic['cl_dd']+nij,mat_project)
    il_ij_fid=np.linalg.inv(cl_ij_fid)
    dl_all=np.zeros([npar,nl,nel,nel])
    #fnl
    dic=read_cls_class('outputs_'+rname+'/run_mfnlcl.dat'); cl_ij_mfnl=dic['cl_dd']
    dic=read_cls_class('outputs_'+rname+'/run_pfnlcl.dat'); cl_ij_pfnl=dic['cl_dd']
    dl_all[0,:,:,:]=transform_matrix((cl_ij_pfnl-cl_ij_mfnl)/(2*0.5),mat_project)

    fisher_prior=np.diag(1./(np.array([1E6]))**2)

    for i1 in np.arange(npar) :
        for l in larr :
            dl_all[i1,l,:,:]=np.dot(dl_all[i1,l],il_ij_fid[l])

    fisher=np.zeros([npar,npar])
    for i1 in np.arange(npar) :
        d1=dl_all[i1]
        for i2 in np.arange(npar) :
            d2=dl_all[i2]
            fisher[i1,i2]=np.sum((larr+0.5)*np.array([np.trace(np.dot(d1[l],d2[l])) for l in larr]))
    fisher+=fisher_prior
    if do_print :
        stout=""
        for s in np.sqrt(np.diag(np.linalg.inv(fisher))) :
            stout+="%.3lE "%s
        print stout

    return fisher

ipars={'mnu':0,'och2':1,'hh':2,'obh2':3,'ns':4,'A_s':5,'lmcb':6,'etab':7,'w0':8}
labpar={'mnu':'$\\Sigma m_\\nu$','och2':'$\\omega_c$','hh':'$h$','obh2':'$\\omega_b$','ns':'$n_s$',
        'A_s':'$A_s$','lmcb':'$\\log_{10}M_c$','etab':'$\\eta_b$','w0':'$w$'}
names=['mnu','och2','hh','obh2','ns','A_s','lmcb','etab','w0']
par0={'mnu':60.,'och2':0.1197,'hh':0.67,'obh2':0.02222,'ns':0.96,'A_s':2.19,'lmcb':14.08,'etab':0.5,'w0':-1.}
def plot_fisher_ll(params,fishers,plotpars,labels,fname_out=None) :

    nfish=len(fishers)
    iplot=np.array([ipars[k] for k in params])
    npar=len(iplot)
                
    fig=plt.figure(figsize=(10,9))
    plt.subplots_adjust(hspace=0,wspace=0)
    for i in np.arange(npar) :
        i_col=i
        for j in np.arange(npar-i)+i :
            i_row=j
            i_plot=i_col+(npar-1)*(i_row-1)+1

            ax=None
            if i!=j :
                ax=fig.add_subplot(npar-1,npar-1,i_plot)
                sig0_max=0
                sig1_max=0
                for i_f in np.arange(nfish) :
                    i1=iplot[i]
                    i2=iplot[j]
                    covar=np.zeros([2,2])
                    covar_full=np.linalg.inv(fishers[i_f])
                    covar[0,0]=covar_full[i1,i1]
                    covar[0,1]=covar_full[i1,i2]
                    covar[1,0]=covar_full[i2,i1]
                    covar[1,1]=covar_full[i2,i2]
                    sig0=np.sqrt(covar[0,0])
                    sig1=np.sqrt(covar[1,1])

                    if sig0>=sig0_max :
                        sig0_max=sig0
                    if sig1>=sig1_max :
                        sig1_max=sig1

                    w,v=np.linalg.eigh(covar)
                    angle=180*np.arctan2(v[1,0],v[0,0])/np.pi
                    a_1s=np.sqrt(2.3*w[0])
                    b_1s=np.sqrt(2.3*w[1])
                    a_2s=np.sqrt(6.17*w[0])
                    b_2s=np.sqrt(6.17*w[1])

                    centre=np.array([par0[names[i1]],par0[names[i2]]])
                    e_1s=Ellipse(xy=centre,width=2*a_1s,height=2*b_1s,angle=angle,
                                 facecolor=plotpars['fc'][i_f],linewidth=plotpars['lw'][i_f],
                                 linestyle="solid",edgecolor=plotpars['lc'][i_f],
                                 alpha=plotpars['alpha'][i_f])
                    ax.add_artist(e_1s)
                    ax.set_xlim([centre[0]-1.75*sig0_max,centre[0]+1.75*sig0_max])
                    ax.set_ylim([centre[1]-1.75*sig1_max,centre[1]+1.75*sig1_max])
                    ax.set_xlabel(labpar[names[i1]],fontsize=FS+2)
                    ax.set_ylabel(labpar[names[i2]],fontsize=FS+2)
                ax.xaxis.set_major_formatter(formatter)
                ax.yaxis.set_major_formatter(formatter)
                for label in ax.get_yticklabels():
                    label.set_fontsize(FS-6)
                for label in ax.get_xticklabels():
                    label.set_fontsize(FS-6)
                if npar==2 :
                    leg_items=[]
                    for il in np.arange(len(labels)) :
                        leg_items.append(plt.Line2D((0,1),(0,0),color=plotpars['lc'][il],
                                                    linestyle='solid',linewidth=plotpars['lw'][il]))
                    ax.legend(leg_items,labels,loc='upper right',frameon=False,fontsize=FS,ncol=2,labelspacing=0.1)
            if ax!=None :
                if i_row!=npar-1 :
                    ax.get_xaxis().set_visible(False)
                else :
                    plt.setp(ax.get_xticklabels(),rotation=45)
                if i_col!=0 :
                    ax.get_yaxis().set_visible(False)
                if i_col==0 and i_row==0 :
                    ax.get_yaxis().set_visible(False)
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])

    if npar!=2 :
        ax=fig.add_subplot(npar-1,npar-1,2)
        ax.set_xlim([-1,1])
        ax.set_ylim([-1,1])
        for il in np.arange(len(labels)) :
            ax.plot([-1,1],[-3,-3],color=plotpars['fc'][il],linestyle='solid',linewidth=2,
                    label=labels[il])
        ax.legend(loc='upper left',frameon=False,fontsize=FS,ncol=2)
        ax.axis('off')

    if fname_out!=None :
        plt.savefig(fname_out,bbox_inches='tight')
#        e_2s=Ellipse(xy=centre,width=2*a_2s,height=2*b_2s,angle=angle,
#                     facecolor=fc[i],linewidth=lw[i],linestyle="dashed",edgecolor=lc[i])

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
