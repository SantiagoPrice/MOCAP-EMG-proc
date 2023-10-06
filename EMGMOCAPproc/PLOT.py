# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 18:12:09 2023

@author: UTAIL
"""
import sys
EMGMOCAPpt =r"C:\Users\UTAIL\OneDrive\Documents\GitHub\MOCAP-EMG-proc"
if not sys.path.count(EMGMOCAPpt):
    sys.path.append(EMGMOCAPpt)
from EMGMOCAPproc import EMG , DEBUG  

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.patches import Patch
import scipy
import tabulate
import time

def comparingAngle(MC_EMG,angle="y",frame_ref="head_abs", participant = "" , data_filtered=False,save=False,check_means=False):
    
    angles = ["y","r","p"]
    titles = ["yaw" , "roll" , "pitch"]
    index=angles.index(angle)
    fig=plt.figure(frame_ref + " " + titles[index] + participant)
    legends=[]
    Xl = r'time [s]'
    
    
    for trial in MC_EMG:
        neu_angs=trial.get_mean_ang()
        offset=neu_angs["value"][0,index]
        mean_angs=trial.get_mean_ang(neutral=False)
        
        Yl = r'{} angle [deg]'.format(titles[index]) 
        if data_filtered:
            RPY=trial.RPY_filtered()[frame_ref]
            plt.title("{} comparisson: {} filtered".format(titles[index],frame_ref))
        else:
            RPY = trial.RPY[frame_ref]
            
            plt.title("{} comparisson for {}: {}".format(titles[index],participant,frame_ref))
        time= np.arange(RPY.shape[1])/trial.sfmc
        Tmax=max(time)
        plt.plot(time,RPY.T[:,index]-offset)
        plt.xlim([0,Tmax])
        
        if check_means:
            for lims in neu_angs["rng"]:
                    plt.axvline(x=lims[0],color="g")
                    plt.axvline(x=lims[1],color="r")
            for angs in mean_angs.values():
                    y=angs["value"][0,index]-offset
                    for lims in angs["rng"]:
                        xmin , xmax = lims
                        plt.axhline(y=y,xmin=xmin/Tmax,xmax=xmax/Tmax,color="b")
        legends.append(trial.label)
    
    plt.axhline(y=0, linestyle='--', color='r')
    plt.grid("on")
    plt.xlabel(Xl)
    plt.ylabel(Yl)
    plt.legend(legends)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    
    if save:
        plt.savefig(r"Plots/{} {} comparisson_ {}.pdf".format(participant,titles[index],frame_ref))
        plt.close(fig)

def ang_stats(part_trials , ref = "head_rel", save = False ):

    
    motions = part_trials[0].mot
    
    cond = ["Low Stiff Part1","Low Stiff Part2","Medium Stiff Part1","Medium Stiff Part2","High Stiff Part1","High Stiff Part1"]
    planes = ["s","l","r"]
    if len(part_trials)==8:
        cond=["Free Part1","Free Part2"]+cond
        
    fig = plt.figure(f"Ang stats {part_trials[0].label[:3]}: {ref}")
    
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Angles for diferent conditions" + part_trials[0].label[:3])
    
    
    ax.set_xticks(np.arange(len(motions)),motions)
    ax.set_yticks([-40,-29,-13,-9,9,13,29,40])
    color_wheel = mcolors.get_named_colors_mapping()
    color_key = list(color_wheel.keys())
    col = [color_key[i*50] for i in range(len(cond))]
    bar_width = 1/(len(part_trials)*2)
              
    for i , mot in enumerate(motions):
        index= planes.index(mot[0])
        means_stds=np.empty((2,0))
        #print(mot , index)
        # Pick the statistics values for each condition
        for trial in part_trials:
            
            offset=trial.get_mean_ang()["value"][:,index]
            
            angs=trial.get_mean_ang(neutral=False,Ref = ref)
            if  list(angs.keys()).count(mot):
                angs=angs[mot]["value"][:,index]
                aux = np.array([[angs[0]-offset[0]],[offset[1]+angs[1]]])
            else:
                aux=np.zeros((2,1))
            means_stds=np.hstack((means_stds,aux))
        
        # Calculate the x positions for the bars
        x = np.linspace(0,len(cond)*bar_width,(len(cond))) + i
        
        # Plot the means as bars
        ax.bar(x, means_stds[0,:], yerr=means_stds[1,:], align='center', alpha=1, width=bar_width, color=col[:len(cond)], label=cond)
        #ax.set_ylim([-6*10**-5, 6*10**-5])
    plt.grid("on")
    if save:
        plt.savefig(r"Plots/{} angle_stats for Head_rel.pdf".format(participant))
        plt.close(fig)

def EMG_plot1p(part_conds,phase="full_cicle",save=False,debug=False, chans=[0,8,9,10], met="mean"):
    muscles = ['(R) Sternocleidomastoid', '(L) Sternocleidomastoid','(R) Para spinal','(L) Para spinal']
    #print(part_conds[0])
    motions = part_conds[0].mot
    motions.sort(key=crit)
    part_N= part_conds[0].label
    #print(f" #printing {part_N}")
    if len(part_conds) == 6:
        cond_lb = ['LS1', 'LS2', 'MS1', 'MS2', 'HS1', 'HS2']
        jcond_lb= ['LS', 'MS', 'HS']
    else:
        cond_lb = ['F1', 'F2', 'LS1', 'LS2', 'MS1', 'MS2', 'HS1', 'HS2']
        jcond_lb= ['F','LS', 'MS', 'HS']
    
   #print(part_N[:3])
    EMG_table = []
    fig = plt.figure(f"EMG_{met}_{part_N} all trials. Phase {phase}.")
     
    print(70*"_"+f"Muscle activity of the participant {int(part_N[1:4])}"+"_"*70)
    for m_n , muscle in enumerate(muscles):
            #print(f"Processsing {muscle}")
            ax = fig.add_subplot(len(muscles),1,m_n+1)
            ax.set_title(part_N[:-4]+ ": " + muscle + " " + phase,fontsize = "xx-large")
            ax.set_xticks(np.arange(len(motions)),motions)
            col_idx = np.linspace(0,255,len(jcond_lb)).astype(int)
            col = list(mpl.cm.gist_rainbow(col_idx))
            patches= [Patch(color=cl , label=cnd) for cl , cnd in zip(col,jcond_lb)]
            
            # Pick the statistics values for each condition
            Ncond=len(jcond_lb)    
            
            means_t=np.zeros((Ncond*2,len(motions)))
            stds_t=np.zeros((Ncond*2,len(motions)))
            means=np.empty((0,len(motions)))
            stds=np.empty((0,len(motions)))
            
            
            max_EMG_val_ind=np.array([EMG.max_EMG(part_conds[-i],channels=chans) for i in range(1,3)])
            
            max_EMG_val=max_EMG_val_ind[:,0,:]
            max_EMG_ind=max_EMG_val_ind[:,1,:]
            max_EMG_abs=max_EMG_val.max(axis=0)
           
            for n_cond , cond_dat in enumerate(part_conds):
                
                means_stds=EMG.EMG_stats(cond_dat,phase,Norm=max_EMG_abs[[m_n]],channels=[chans[m_n]])[:,0,:]

                for nm_loc, m in enumerate(cond_dat.mot):
                    nm_glb=motions.index(m)
                    means_t[n_cond,nm_glb] = means_stds[nm_loc,0]
                    stds_t[n_cond,nm_glb] = means_stds[nm_loc,1]
                    
                    if m =="r29" and (n_cond==0 or n_cond==5) and m_n==1:
                        a=["ls","hs"]
                        
                       #print(a[n_cond==0])
                       #print(means_stds[nm_loc,0])
                    
                    if debug:
                        if m_n ==1 and n_cond==0:
                                pass
                                #print(f"Mean activity of {muscle} for {m}")
                                #print(means_stds[nm_loc,0])
            for condf in jcond_lb:
                trial1=cond_lb.index(f"{condf}1")
                trial2=cond_lb.index(f"{condf}2")
                means_t[trial2,means_t[trial2]==0] = means_t[trial1,means_t[trial2]==0]
                means_t[trial1,means_t[trial1]==0] = means_t[trial2,means_t[trial1]==0]
                
                    
                #means=np.vstack((means,means_t[trial2,:]))
                #stds=np.vstack((stds,stds_t[trial2,:]))
                means=np.vstack((means,means_t[trial1:trial2+1,:].mean(axis = 0)))
                stds=np.vstack((stds,stds_t[trial1:trial2+1,:].mean(axis = 0)))
               
                
                #print(condf)
                #print(means_t[trial1,:])
                #print(means_t[trial2,:])
                #print(means[-1,:])
            minimal = False
            if minimal:
                L40s=[motions.index("l40")]+[motions.index("l-40")]
                for n_cond , cond_lab in enumerate(jcond_lb): 
                    EMG_table.append([cond_lab]+["{:.4f} +-{:.4f}".format(m,s) for m , s in zip(means[n_cond,L40s],stds[n_cond,L40s])])
                    heads=["l40_left","l40_right"]
            else:
                for n_cond , cond_lab in enumerate(jcond_lb):    
                    EMG_table.append([cond_lab]+["{:.4f} +-{:.4f}".format(m,s) for m , s in zip(means[n_cond],stds[n_cond])])
                    heads=motions
            if "Stern" in muscle:
                print(muscle)
                print(tabulate.tabulate(EMG_table, headers = heads, tablefmt="fancy_grid"))
            EMG_table=[]
            means=means.T.reshape(-1)
            stds=stds.T.reshape(-1)
            ##print(f"means: {means}")
            # Calculate the x positions for the bars
            x = np.array([np.linspace(0,0.5,Ncond) + i for i in range(len(motions))]).reshape(-1)
  
            # Plot the means as bars
            ##print(abs(means))
            ax.bar(x, abs(means), yerr=stds, align='center', alpha=0.7, width=0.5/Ncond, color=col*Ncond)
            if m_n==2:
                ax.legend(handles=patches,bbox_to_anchor=(1, 0.))
            plt.grid("on")
            ax.set_ylim([0, max(abs(means))*1.1])
           # ax.update(hspace=0.5)
    time.sleep(4)
    mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()  
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.50)       
    if save:
                plt.savefig(r"Plots/{} EMG profile {}.pdf".format(part_N,phase))

def crit(a):
    return ord(a[0])*100+int(a[1:])

def EMG_plotmultip(parts,phase="full_cicle",save=False,debug=False, Norm="manual", channels=[0,8,9,10], met="mean" ,style=None):
    muscles = ['(R) Sternocleidomastoid', '(L) Sternocleidomastoid','(R) Para spinal','(L) Para spinal']
    Nparts= len(parts)
    
    motions = parts["P 3"][0].mot
    motions.sort(key=crit)
    Nmot= len(motions)
    cond_lb = ['LS1', 'LS2', 'MS1', 'MS2', 'HS1', 'HS2']
    jcond_lb= ['LS', 'MS', 'HS']
    
    custNorm={}
    
    exc_part=[]
    
    
    Max_EMGs_manual={}

    Max_EMGs_manual.update({"P 3":np.array([9.33e-5,1.87e-3,6.87e-6,5.11e-6])})
    Max_EMGs_manual.update({"P 4":np.array([1.34e-4,2.4e-4,1.14e-5,1.083e-5])})
    Max_EMGs_manual.update({"P 5":np.array([6.15e-5,3.5e-5,2.27e-5,2.76e-5])}) #Very spiky
    Max_EMGs_manual.update({"P 6":np.array([1.4e-4,577e-5,1.73e-5,2.2e-5])})
    Max_EMGs_manual.update({"P 8":np.array([2.9e-5,2.67e-5,7.6e-6,8.6e-6])})
    Max_EMGs_manual.update({"P 9":np.array([2.2e-4,2.84e-4,7.54e-6,9.1e-6])})
    Max_EMGs_manual.update({"P10":np.array([3.59e-5,1.03e-5,5.25e-6,1.98e-4])}) #very spiky
    Max_EMGs_manual.update({"P 11":np.array([1.6e-4,1.5e-4,8.86e-6,7.39e-6])})
    
    mot_types={"axial rotation":"r" , "sagital flexion":"s","lateral flexion":"l"}
    figs=[] #Figure container
    
    for m_n , muscle in enumerate(muscles):
            print(muscle)
            #print(f"Processsing {muscle}")
            
            
            col_idx = np.linspace(0,255,len(jcond_lb)).astype(int)
            col = list(mpl.cm.gist_rainbow(col_idx))
            patches= [Patch(color=cl , label=cnd) for cl , cnd in zip(col,jcond_lb)]
            
            # Pick the statistics values for each condition
            Ncond=len(jcond_lb)    
            
            means_t=np.zeros((Ncond*2,Nmot,Nparts))
            stds_t=np.zeros((Ncond*2,Nmot,Nparts))
            means=np.empty((0,len(motions)))
            stds=np.empty((0,len(motions)))
            
            act_part=[] #list with the active participants
            act_part_lab=[] #label list with the active participants
            for  part_lab, part_conds in parts.items():
                
                    
                if part_lab=="P 1" or part_lab=="P 2" or part_lab=="P 7" or part_lab=="P 10":
                    continue
                if part_lab == "P 11":
                    chans=[48,56,57,58]
                else:
                    chans=channels
                
                
                #max_EMG_val_ind=np.array([EMG.max_EMG(part_conds[i],channels=chans) for i in range(len(part_conds))])
                max_EMG_val_ind=np.array([EMG.max_EMG(part_conds[-i],channels=chans) for i in range(1,3)])
                
                max_EMG_val=max_EMG_val_ind[:,0,:]
                max_EMG_ind=max_EMG_val_ind[:,1,:]
                if Norm =="manual":
                    max_EMG_abs=Max_EMGs_manual[part_lab]
                else:
                    max_EMG_abs=max_EMG_val.max(axis=0)
                
                
                
                #max_EMG=EMG.max_EMG(part_conds[-1])
                npa=int(part_lab[-2:])-1
                act_part.append(npa)
                act_part_lab.append(part_lab)
                # if m_n==1:
                #     print(part_lab)
                #     print(npa)
                if debug:
                    max_EMG_abs_ind = max_EMG_val[:,m_n].argmax(axis=0)
                    MaxEMG_time= max_EMG_ind[max_EMG_abs_ind,m_n]/part_conds[0].sfemg
                    
                    if part_lab in custNorm.keys():
                        max_EMG_abs_ind = 0
                        MaxEMG_time= 0
       
                    max_EMG_abs_ind+=1
                    max_EMG_abs_ind*=-1
                    
                    flab="EMG vs time "+part_conds[max_EMG_abs_ind].label
                    
                    if plt.get_figlabels().count(flab):
                        figEMG=plt.figure(flab)
                    else:
                        figEMG=DEBUG.EMGvsTime(part_conds[max_EMG_abs_ind],filt=True,shaded=False,RPY_add=False,channels=chans,Filt_type=2)
                        
                    axEMGmax=figEMG.get_axes()[m_n]
                    axEMGmax.axvline(x=MaxEMG_time, ymin=0, ymax=1,linestyle="--",color="r")
                    
                    # if len(part_conds) == 8:
                    #     continue
                    #     max_EMG_abs_ind-=2
                    
                    print("Abs EMG max for {} at {} seconds".format(part_conds[max_EMG_abs_ind].label , MaxEMG_time))
                    continue
                    
                if len(part_conds)==8:
                    pcs= part_conds[2:]
                else:
                    pcs=part_conds
                    
                for n_cond , cond_dat in enumerate(pcs):
                    if part_lab == "P 2":
                        pass
                       #print(part_lab + cond_lb[n_cond])
                       #print(n_cond)
                       #print(Norm[npa])
                    
                    
                    means_stds_1p=EMG.EMG_stats(cond_dat,phase , Norm=max_EMG_abs[[m_n]] , channels=[chans[m_n]] , metric=met)[:,0,:]
                  
                    
                    for nm_loc, m in enumerate(cond_dat.mot):
                        nm_glb=motions.index(m)
                        means_t[n_cond,nm_glb,npa] = means_stds_1p[nm_loc,0]
                    
                        
                        stds_t[n_cond,nm_glb,npa] = means_stds_1p[nm_loc,1]
                        
                        
                for condf in jcond_lb:
                    trial1=cond_lb.index(f"{condf}1")
                    trial2=cond_lb.index(f"{condf}2")
                    means_t[trial2,means_t[trial2,:,npa]==0,npa] = means_t[trial1,means_t[trial2,:,npa]==0,npa]
                    means_t[trial1,means_t[trial1,:,npa]==0,npa] = means_t[trial2,means_t[trial1,:,npa]==0,npa]
                
            if debug:
                continue
                #means=np.vstack((means,means_t[trial2,:]))
                #stds=np.vstack((stds,stds_t[trial2,:]))
            pGroups=means_t[:,:,act_part].reshape(Ncond,2,Nmot,len(act_part)).mean(axis = 1)[np.arange(3)!=1]
            
            #pTest=scipy.stats.wilcoxon(pGroups[0],pGroups[1],axis=-1 , alternative="greater")
            pTest=[scipy.stats.wilcoxon(pGroups[0,mvnt,:],pGroups[1,mvnt,:],axis=-1 , alternative="greater") for mvnt in range(Nmot)]
            
            
            #pTest=scipy.stats.friedmanchisquare(list(pVals[0]),list(pVals[1]),list(pVals[2]))
            mr29=means_t.reshape(2,Ncond,Nmot,Nparts).mean(axis = 0)
            iii=motions.index("r-29")
            #print(muscles[m_n])
            #print(mr29[:,iii,:])
            
            #Getting the mean from each pair of identical conditions
            
            means=means_t[:,:,act_part].reshape(Ncond,2,Nmot,len(act_part)).mean(axis = 1)
          
          
            # Plot the means as bars
            ##print(abs(means))
            
            if m_n == 0:    
                exc_part=set([f"P {i+1}" for i in range(Nparts)]) ^ set(act_part_lab) 
            
            if style==None:
                fig=plt.figure(f"EMG_{met}_{style} group: {phase} phase")
                #Getting the means mean and stds
                stds=stds.means(axis = -1)
                
                means=means.means(axis = -1)
                #stds=stds_t.reshape(2,Ncond,Nmot,Nparts).mean(axis = 0).mean(axis = -1)
                
                
                
                means=means.T.reshape(-1)
                stds=stds.T.reshape(-1)
                ##print(f"means: {means}")
                # Calculate the x positions for the bars
                x = np.array([np.linspace(0,0.5,Ncond) + i for i in range(len(motions))]).reshape(-1)
                
                ax = fig.add_subplot(len(muscles),1,m_n+1)
                ax.set_title("All parts:"  + muscle + " " + phase,fontsize = "xx-large")
                ax.set_xticks(np.arange(len(motions)),motions)
                ax.bar(x, abs(means), yerr=stds, align='center', alpha=0.7, width=0.5/Ncond, color=col*Ncond)
                if m_n==2:
                    ax.legend(handles=patches,bbox_to_anchor=(1.2, 0.))
                plt.grid("on")
                ax.set_ylim([0, max(abs(means))*1.1])
                #for npv , pv in enumerate(pTest.pvalue):
                for npv , pv in enumerate(pTest):
                    barplot_annotate_brackets(npv*3 , npv*3+2, f"p = {pv.pvalue:.2f}" , x , abs(means),ax)
                
            elif style=="box":
                
                for mt_name , mt_key in mot_types.items():
                    mots=[m for m in motions if m[0] == mt_key]
                    
                    fig=plt.figure(f"EMG_{met}_{style} group for {mt_name}: {phase} phase")
                    if m_n == 0:
                        fig.suptitle(f"EMG_{met}_{style} group for {mt_name}: {phase} phase("+",".join(exc_part)+" are excluded)",fontsize = "xx-large")
                        fig.tight_layout(pad=3, w_pad=0.5, h_pad=0.50) 
                        figs.append(fig)
                        
                        
                    for nmt , mt in enumerate(mots):
                        ax = fig.add_subplot(len(muscles),len(mots),(m_n*len(mots)+nmt+1))
                        ax.set_title(muscle + " " + mt,fontsize = "xx-large")
                        
                        plt.boxplot(means[:,nmt,:].T,labels=jcond_lb)
                        heights=means[:,nmt,:].max(axis=-1)
                        pv=pTest[nmt]
                        barplot_annotate_brackets(0, 2, f"p = {pv.pvalue:.2f}" , np.arange(1,4) , heights,ax)
            
            elif style=="lines":
                for mt_name , mt_key in mot_types.items():
                    mots=[m for m in motions if m[0] == mt_key]
                    
                    fig=plt.figure(f"EMG_{met}_{style} group for {mt_name}: {phase} phase")
                    if m_n == 0:
                        fig.suptitle(f"EMG_{met}_{style} group for {mt_name}: {phase} phase("+",".join(exc_part)+" are excluded)",fontsize = "xx-large")
                        fig.tight_layout(pad=3, w_pad=0.5, h_pad=0.50) 
                        figs.append(fig)
                        
                        
                    for nmt , mt in enumerate(mots):
                        ax = fig.add_subplot(len(muscles),len(mots),(m_n*len(mots)+nmt+1))
                        ax.set_title(muscle + " " + mt,fontsize = "xx-large")
                        ax.set_xticks(np.arange(3),jcond_lb)
                        
                        legs=[f"P{ap+1}" for ap in act_part]
                        ax.plot(means[:,nmt,:],label=legs)
                        if m_n == 0 and nmt==0:
                            plt.legend()
                        
               
            
            
            
           # ax.update(hspace=0.5)
    
    #mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()  
         
    # print("Means matrix")
    # print(means_t[:,:,act_part])
    if save:
                plt.savefig(r"./Plots/9302023/EMG_{}_{} group {} phase.pdf".format(met,style,phase))

    
def barplot_annotate_brackets(num1, num2, data, center, height, ax, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh
    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    ax.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    ax.text(*mid, text, **kwargs)

# def filter_data(MC_EMG):
    
#     for trial in part_MC_EMG:
#         plt.figure()
#         RPY = trial.RPY["body_abs"]
#         # Select the desired array to filter and plot (e.g., index 0 for the first array)
#         # Define the filter parameters
#         order = 4  # Filter order
#         cutoff_freqs = np.linspace(0.001, 0.01, num=10)  # Cutoff frequencies for filtering
#         # Define the vertical shift amount for visualization
#         shift_amount = 5.0
#         # Apply low-pass filter to the curve with increasing cutoff frequencies
#         for i, cutoff_freq in enumerate(cutoff_freqs):
#             # Design the Butterworth low-pass filter
#             b, a = butter(order, cutoff_freq, btype='low', analog=False, output='ba')
#             # Apply the filter to the curve
#             filtered_curve = filtfilt(b, a, RPY[2, :])
#             # Calculate the vertical shift value
#             shift_value = i * shift_amount
#             # Plot the filtered curve with vertical shift
#             plt.plot(filtered_curve + shift_value)
            
#         mng = plt.get_current_fig_manager()
#         mng.full_screen_toggle()  
#         plt.xlabel('Column')
#         plt.ylabel('Value')
#         plt.title('Low-Band Filtered Curves with Vertical Shift: {}'.format(trial.label))
#         plt.show()


