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
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from ezc3d import c3d
import scipy
import tabulate
import time
import os
import csv
 

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
    # if save:
    #     plt.savefig(r"Plots/{} angle_stats for Head_rel.pdf".format(participant))
    #     plt.close(fig)
        
def ANG_plotmultip(parts,save=False,debug=False, met="mean" ,style=None ,ref="head_rel"):
    Nparts= len(parts)
    
    motions = parts["P 1"][0].mot
    motions.sort(key=crit)
    Nmot= len(motions)
    planes = ["s","l","r"]
    cond_lb = ['LS1', 'LS2', 'MS1', 'MS2', 'HS1', 'HS2']
    jcond_lb= ['LS', 'MS', 'HS']
    
    if debug:
        print(motions)
    custNorm={}
    
    exc_part=[]
    
    

    
    mot_types={"axial rotation":"r" , "sagital flexion":"s","lateral flexion":"l"}
    figs=[] #Figure container
    Ang_table=[] #Table container
    
 
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
      
        print(part_lab)    
        if part_lab=="P 1" or part_lab=="P 2"or part_lab=="P 10" or part_lab=="P 8":#  or part_lab=="P 5"  or part_lab=="P 7"
             continue
       
        
        
        
        #max_EMG=EMG.max_EMG(part_conds[-1])
        npa=int(part_lab[-2:])-1
        act_part.append(npa)
        act_part_lab.append(part_lab)
        # if m_n==1:
        #     print(part_lab)
        #     print(npa)

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
            
            means_stds_1p=cond_dat.get_mean_ang(neutral=False,Ref =ref)
            neu_angs=cond_dat.get_mean_ang(neutral=True,Ref =ref)                             
            
            for nm_loc, m in enumerate(cond_dat.mot):
                nm_glb=motions.index(m)
                index=planes.index(m[0])
                means_t[n_cond,nm_glb,npa] = means_stds_1p[m]["value"][0,index]  - neu_angs["value"][0,index]      
                stds_t[n_cond,nm_glb,npa] = means_stds_1p[m]["value"][1,index]
                
                
        for condf in jcond_lb:
            trial1=cond_lb.index(f"{condf}1")
            trial2=cond_lb.index(f"{condf}2")
            means_t[trial2,means_t[trial2,:,npa]==0,npa] = means_t[trial1,means_t[trial2,:,npa]==0,npa]
            means_t[trial1,means_t[trial1,:,npa]==0,npa] = means_t[trial2,means_t[trial1,:,npa]==0,npa]
        
        
    pGroups=means_t[:,:,act_part].reshape(Ncond,2,Nmot,len(act_part)).mean(axis = 1)[np.arange(3)!=1]
    if debug:
            print(pGroups)
   
            print(act_part)
    #pTest=scipy.stats.wilcoxon(pGroups[0],pGroups[1],axis=-1 , alternative="greater")
    pTest=[scipy.stats.wilcoxon(pGroups[0,mvnt,:],pGroups[1,mvnt,:],axis=-1 , alternative="less") for mvnt in range(Nmot)]
    #print(pTest)
    pTest2=[scipy.stats.wilcoxon(pGroups[0,mvnt,:],pGroups[1,mvnt,:],axis=-1 , alternative="two-sided") for mvnt in range(Nmot)]
    
    #pTest=scipy.stats.friedmanchisquare(list(pVals[0]),list(pVals[1]),list(pVals[2]))
    #print(muscles[m_n])
    #print(mr29[:,iii,:])
    
    #Getting the mean from each pair of identical conditions
    
    means=means_t[:,:,act_part].reshape(Ncond,2,Nmot,len(act_part)).mean(axis = 1)
    stds=means_t[:,:,act_part].reshape(Ncond,2,Nmot,len(act_part)).std(axis = 1)
  
    # Plot the means as bars
    ##print(abs(means))
    
    exc_part=set([f"P {i+1}" for i in range(Nparts)]) ^ set(act_part_lab) 
    
    if style==None:
        fig=plt.figure(f"ANG_{met}_{style} group")
        #Getting the means mean and stds
        stds=stds.means(axis = -1)
        
        means=means.means(axis = -1)
        #stds=stds_t.reshape(2,Ncond,Nmot,Nparts).mean(axis = 0).mean(axis = -1)
        
        
        
        means=means.T.reshape(-1)
        stds=stds.T.reshape(-1)
        ##print(f"means: {means}")
        # Calculate the x positions for the bars
        x = np.array([np.linspace(0,0.5,Ncond) + i for i in range(len(motions))]).reshape(-1)
        
        ax = fig.add_subplot()
        ax.set_title("All parts angle",fontsize = "xx-large")
        ax.set_xticks(np.arange(len(motions)),motions)
        ax.bar(x, abs(means), yerr=stds, align='center', alpha=0.7, width=0.5/Ncond, color=col*Ncond)
        ax.legend(handles=patches,bbox_to_anchor=(1.2, 0.))
        plt.grid("on")
        ax.set_ylim([0, max(abs(means))*1.1])
        #for npv , pv in enumerate(pTest.pvalue):
        for npv , pvs in enumerate(zip(pTest,pTest2)):
            pv1=pvs[0]
            pv2=pvs[1]
            barplot_annotate_brackets(npv*3 , npv*3+2, f"p =st: {pv1.pvalue:.2f}, dt: {pv2.pvalue:.2f}" , x , abs(means),ax)
        
    elif style=="box":
        
        for mt_name , mt_key in mot_types.items():
            mots=[m for m in motions if m[0] == mt_key]
            print(mots)
            fig=plt.figure(f"Ang displ_{met}_{style} group for {mt_name}")
            
            fig.suptitle(f"Ang displ_{met}_{style} group for {mt_name} ("+",".join(exc_part)+" are excluded)",fontsize = "xx-large")
            fig.tight_layout(pad=3, w_pad=0.5, h_pad=0.50) 
            figs.append(fig)
            
                
            for mt_ind , mt in enumerate(mots):
                nmt=motions.index(mt)
                ax = fig.add_subplot(1,len(mots),(mt_ind+1))
                ax.set_title( mt,fontsize = "xx-large")
                
                plt.boxplot(means[:,nmt,:].T,labels=jcond_lb)
                heights=means[:,nmt,:].max(axis=-1)
                pv1=pTest[nmt]
                pv2=pTest2[nmt]
                barplot_annotate_brackets(0, 2, f"p =st: {pv1.pvalue:.2f}, dt: {pv2.pvalue:.2f}"  , np.arange(1,4) , heights,ax)
                
    elif style=="box_minimal":
        
        for mt_name , mt_key in mot_types.items():
            mots=[m for m in motions if m[0] == mt_key]
            print(mots)
            if mt_key == "s":
                side= ["back" , "front"]
            else:
                side=["right" , "left"]
                
            fig=plt.figure(f"EMG_{met}_{style} group for {mt_name}2")
            
            fig.suptitle(f"Ang_{met}_{style} group for {mt_name}: ("+",".join(exc_part)+" are excluded)",fontsize = "xx-large")
            fig.tight_layout(pad=3, w_pad=0.5, h_pad=0.50) 
            figs.append(fig)
            if mt_key == "l":
                aux= [mots[1],mots[-2]]
            elif mt_key == "s":
                aux= [mots[1],mots[-1]]
            else:                            
                aux= [mots[0],mots[-1]]
            mots=aux
            print(mots)
            for mt_ind , mt in enumerate(mots):
                nmt=motions.index(mt)
                ax = fig.add_subplot(1,len(mots),(mt_ind+1))
                
                abs_ang= str(np.abs(int(mt[1:])))
                
                ax.set_title(" " + side[mt_ind] + " " + abs_ang,fontsize = 22)
                
                plt.boxplot(means[[0,-1],nmt,:].T,labels=[jcond_lb[0] , jcond_lb[-1]])
                plt.yticks(fontsize=20)
                plt.xticks(fontsize=20)
                heights=means[[0,-1],nmt,:].max(axis=-1)
                pv1=pTest[nmt]
                pv2=pTest2[nmt]
                
                if pv1.pvalue < 0.05:
                    if pv2.pvalue < 0.05:
                        barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax,col="green")
                    else:
                        barplot_annotate_brackets(0, 1, pv1.pvalue , np.arange(1,3) , heights,ax,col="blue")
                else:
                    if pv2.pvalue < 0.05:
                        barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax,col="red")
                    else:
                        barplot_annotate_brackets(0, 1, r"\noindent HS=LS?: {:.2f}  \\ \\  HS$<$LS?: {:.2f}".format(pv2.pvalue,pv1.pvalue), np.arange(1,4) , heights,ax,barh=0.12)
                # barplot_annotate_brackets(0, 1, f"{pv2.pvalue}" , np.arange(1,3) , heights,ax)
                # if pv2.pvalue < 0.05:
                #     barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax)
    
    
    elif style=="table":
        stds=means.std(axis = -1)
        means=means.mean(axis = -1)
        
        ang_dict=dict()   #Iam using this to get the angles
        mot=["Fflex/Rbend" , "Bflex/LBend"]
        heads=[""]
        val=[["high","low"]]
        val+=[["low","high"]]
        for v in val:
            for magn,m in zip(v,mot):
                heads+=[f"{magn} {m}"]
        
        conds=["LS","MS","HS"]
        
        Ang_table.append(heads)
        for mt_name , mt_key in mot_types.items():
            
            mots=[m for m in motions if m[0] == mt_key]
            ang_ms_std=np.zeros([len(mots),len(conds),2])
            
            ang_dict.update({mt_key:ang_ms_std})
            
            for ncond , cond in enumerate(conds):
                Ang_table.append([" ".join([mt_name,cond])])
                
                
                if mt_key == "s":
                    side= ["back" , "front"]
                else:
                    side=["right" , "left"]

                
                for mt_ind , mt in enumerate(mots):
    
                    nmt=motions.index(mt)
                    #print(nmt)
    

                    
                    
                    #pv1=pTest[nmt]
                    #pv2=pTest2[nmt]
                    #means[0,nmt]
                    Ang_table[-1].append("{:.2f} +-{:.2f}".format(means[ncond,nmt],stds[ncond,nmt]) ) 
                    ang_ms_std[mt_ind,ncond,0]=means[ncond,nmt]
                    ang_ms_std[mt_ind,ncond,1]=stds[ncond,nmt]
                    #Ang_table[-1].append(pv2.pvalue)
                    #Ang_table[-1].append(pv1.pvalue)
        
        
        #print(tabulate.tabulate(Ang_table, headers = heads, tablefmt="fancy_grid")) 
       
        # with open(r"G:/My Drive/PhD/My research/Experiments/EMG and MOCAP measurements/5- 6 bar rigid blockage, 6 users/Plots/table2.csv","w") as f:
        #     writee=csv.writer(f)
        #     writee.writerows(Ang_table)
        return ang_dict
    
    
    elif style=="table_minimal":
        stds=means.std(axis = -1)
        means=means.mean(axis = -1)
        
        heads=[""]
        
        for mot in ["Fflex/Rbend" , "Bflex/LBend"]:
            heads+=[f"{mot}_LS"]+ [f"{mot}_HS"] + ["HS != LS?"] + ["HS > LS?"]
        
        Ang_table.append(heads)
        for mt_name , mt_key in mot_types.items():
            Ang_table.append([mt_name])
            
            mots=[m for m in motions if m[0] == mt_key]
            if mt_key == "s":
                side= ["back" , "front"]
            else:
                side=["right" , "left"]
                
            if mt_key == "l":
                aux= [mots[1],mots[-2]]
            elif mt_key == "s":
                aux= [mots[1],mots[-1]]
            else:                            
                aux= [mots[0],mots[-1]]
            mots=aux
            
            for mt_ind , mt in enumerate(mots):

                nmt=motions.index(mt)
                print(nmt)

                abs_ang= [side[mt_ind]] + [str(np.abs(int(mt[1:])))]
                
                
                pv1=pTest[nmt]
                pv2=pTest2[nmt]
                means[0,nmt]
                [Ang_table[-1].append("{:.2f} +-{:.2f}".format(means[cnd,nmt],stds[cnd,nmt]) ) for cnd in [0,2]]
                Ang_table[-1].append(pv2.pvalue)
                Ang_table[-1].append(pv1.pvalue)
        
        print(tabulate.tabulate(Ang_table, headers = heads, tablefmt="fancy_grid"))  
        with open(r"G:/My Drive/PhD/My research/Experiments/EMG and MOCAP measurements/5- 6 bar rigid blockage, 6 users/Plots/table2.csv","w") as f:
            writee=csv.writer(f)
            writee.writerows(Ang_table)
                
                    
            
        
                
                
    
    elif style=="lines":
        for mt_name , mt_key in mot_types.items():
            mots=[m for m in motions if m[0] == mt_key]
            
            fig=plt.figure(f"Ang_{met}_{style} group for {mt_name}")
            
            fig.suptitle(f"Ang_{met}_{style} group for {mt_name}("+",".join(exc_part)+" are excluded)",fontsize = "xx-large")
            fig.tight_layout(pad=3, w_pad=0.5, h_pad=0.50) 
            figs.append(fig)
                
                
            for mt_ind , mt in enumerate(mots):
                nmt=motions.index(mt)
                ax = fig.add_subplot(1,len(mots),(mt_ind+1))
                ax.set_title( mt,fontsize = "xx-large")
                ax.set_xticks(np.arange(3),jcond_lb)
                
                legs=[f"P{ap+1}" for ap in act_part]
                ax.plot(means[:,nmt,:],label=legs)
                ax.legend()
                    
    elif style=="lines_minimal":
        for mt_name , mt_key in mot_types.items():
            mots=[m for m in motions if m[0] == mt_key]
            
            fig=plt.figure(f"Ang_{met}_{style} group for {mt_name}")

            fig.suptitle(f"Ang_{met}_{style} group for {mt_name} ("+",".join(exc_part)+" are excluded)",fontsize = "xx-large")
            fig.tight_layout(pad=3, w_pad=0.5, h_pad=0.50) 
            figs.append(fig)
                
            aux= [mots[0],mots[-1]]
            mots=aux    
            rng=np.arange(3)!=1
            for mt_ind , mt in enumerate(mots):
                nmt=motions.index(mt)
                ax = fig.add_subplot(1,len(mots),(mt_ind+1))
                ax.set_title(mt,fontsize = "xx-large")
                ax.set_xticks(np.arange(3)[rng],[jcond_lb[0],jcond_lb[2]])
                
                legs=[f"P{ap+1}" for ap in act_part]
                
                ax.plot(means[rng,nmt,:],label=legs)
                ax.legend(loc="upper right")
                
               
            
            
            
           # ax.update(hspace=0.5)
    
    #mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()  
         
    # print("Means matrix")
    # print(means_t[:,:,act_part])
    if save:
                plt.savefig(r"./Plots/9302023/EMG_{}_{}_group.pdf".format(met,style))


def EMG_plot1p(part_conds,phase="full_cicle",save=False,debug=False, chans=[0,8,9,10], met="mean", labs=None):
    #print(part_conds[0])
    motions = part_conds[0].mot
    motions.sort(key=crit)
    part_N= part_conds[0].label
    #print(f" #printing {part_N}")
    
    muscles = ['(R) Sternocleidomastoid', '(L) Sternocleidomastoid','(R) Para spinal','(L) Para spinal']
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
            #max_EMG_abs=np.ones(max_EMG_abs.shape)
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

def EMG_plot1p_EMBC(part_conds,phase="full_cicle",save=False,debug=False, chans=[0,8,9,10], met="mean", labs=None):
    #print(part_conds[0])
    motions = part_conds[0].mot
    motions.sort(key=crit)
    part_N= part_conds[0].label
    #print(f" #printing {part_N}")
    
    muscles = ['(R) Sternocleidomastoid', '(L) Sternocleidomastoid','(R) Para spinal','(L) Para spinal','(R) U Trapezious','(L) U Trapezious']
    if len(part_conds) == 4:
        cond_lb = ['LS1', 'LS2', 'MS1', 'MS2', 'HS1', 'HS2']
        jcond_lb= ["F",'L', 'M', 'H']
    else:
        cond_lb = ['F1', 'F2', 'LS1', 'LS2', 'MS1', 'MS2', 'HS1', 'HS2']
        jcond_lb= ['L', 'M', 'H']
    
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
            
            means=np.zeros((Ncond,len(motions)))
            stds=np.zeros((Ncond,len(motions)))
            
            
            # max_EMG_val_ind=np.array([EMG.max_EMG(part_conds[-i],channels=chans) for i in range(1,3)])
            
            # max_EMG_val=max_EMG_val_ind[:,0,:]
            # max_EMG_ind=max_EMG_val_ind[:,1,:]
            # max_EMG_abs=max_EMG_val.max(axis=0)
            # max_EMG_abs=np.ones(max_EMG_abs.shape)
            for n_cond , cond_dat in enumerate(part_conds):
                
                means_stds=EMG.EMG_stats(cond_dat,phase,Norm=[None],channels=[chans[m_n]])[:,0,:]
                
                
                for nm_loc, m in enumerate(cond_dat.mot):
                    nm_glb=motions.index(m)
                    means[n_cond,nm_glb] = means_stds[nm_loc,0]
                    stds[n_cond,nm_glb] = means_stds[nm_loc,1]
                        
                       #print(a[n_cond==0])
                       #print(means_stds[nm_loc,0])
                    
                    if debug:
                        if m_n ==1 and n_cond==0:
                                pass
                                #print(f"Mean activity of {muscle} for {m}")
                                #print(means_stds[nm_loc,0])
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

def EMG_plotmultip_3phs(partss,savee=False, Normm="manual", channelss=[0,8,9,10], mett="mean"):
    phs=["aproaching","holding","recovery"]
    for ph in phs:
        if ph == phs[-1] and savee:
            is_saved=True
        else:
            is_saved=False
        EMG_plotmultip(parts=partss ,phase=ph,save=is_saved, Norm=Normm, channels=channelss, met=mett,style="box_minimal_3phases")



def EMG_plotmultip(parts,phase="holding",save=False,debug=False, Norm="manual", channels=[0,8,9,10], met="mean" ,style=None , mGroup=None):
    muscles = ['(R) Sternocleidomastoid', '(L) Sternocleidomastoid','(R) Para spinal','(L) Para spinal']
    muscles = ['right STR', 'left STR','right SPL','left SPL']
    Nparts= len(parts)
    
    motions = parts["P 3"][0].mot
    motions.sort(key=crit)
    Nmot= len(motions)
    nMusc= len(muscles)
    cond_lb = ['LS1', 'LS2', 'MS1', 'MS2', 'HS1', 'HS2']
    jcond_lb= ['LS', 'MS', 'HS']
    
    means=dict() # Container with means EMG values of each INVOLVED participant
    stds=dict() # Container with stds EMG values of each INVOLVED participant
    
    print("Note: intermediate stiffness is not being processed")
    
    print(motions)
    custNorm={}
    
    exc_part=[]
    
    
    
    # Defining normalization critera
    
    Max_EMGs_manual={}
    Max_EMGs_manual.update({"P 1":np.array([4.2e-5,2e-5,1.71e-5,1e-5])})
    Max_EMGs_manual.update({"P 2":np.array([7.5e-5,9.6e-5,9e-6,1.3e-5])})
    Max_EMGs_manual.update({"P 3":np.array([9.33e-5,9.33e-5,6.87e-6,5.11e-6])})
    Max_EMGs_manual.update({"P 4":np.array([1.34e-4,1e-4,1.25e-5,7e-6])})
    Max_EMGs_manual.update({"P 5":np.array([6.15e-5,3.5e-5,2.27e-5,2.76e-5])}) #Very spiky
    Max_EMGs_manual.update({"P 6":np.array([1.4e-4,5.7e-5,1.73e-5,2.2e-5])})
    Max_EMGs_manual.update({"P 7":np.array([5e-4,5e-5,5e-5,3.42e-5])})
    Max_EMGs_manual.update({"P 8":np.array([2.9e-5,2.67e-5,7.6e-6,8.6e-6])})
    Max_EMGs_manual.update({"P 9":np.array([2.2e-4,2.84e-4,7.54e-6,9.1e-6])})
    Max_EMGs_manual.update({"P 10":np.array([3.59e-5,3.8e-5,5.25e-6,1.98e-4])}) #very spiky
    Max_EMGs_manual.update({"P 11":np.array([1.6e-4,1.5e-4,8.86e-6,7.39e-6])})
    
    
    
    
    mot_types={"axial rotation":"r" , "sagital flexion":"s","lateral flexion":"l"}
    figs=[] #Figure container
    EMG_table=[] #Table container
    
    col_idx = np.linspace(0,255,len(jcond_lb)).astype(int)
    col = list(mpl.cm.gist_rainbow(col_idx))
    patches= [Patch(color=cl , label=cnd) for cl , cnd in zip(col,jcond_lb)]
    
    # Pick the statistics values for each condition
    Ncond=len(jcond_lb)  
    
    
    means_t=np.zeros((Ncond*2,Nmot,Nparts,nMusc))
    stds_t=np.zeros((Ncond*2,Nmot,Nparts,nMusc))
    #means=np.empty((0,len(motions)))
    #stds=np.empty((0,len(motions)))
    for m_n , muscle in enumerate(muscles):
            print(muscle)
            #print(f"Processsing {muscle}")
            
            
              
            
            
            
            act_part=[] #list with the active participants
            act_part_lab=[] #label list with the active participants
            for  part_lab, part_conds in parts.items():
              
                
              #Filtering participants
                    
                if part_lab=="P 10" or part_lab=="P 8" or  part_lab=="P 1":# or part_lab=="P 7" or part_lab=="P 2" or part_lab=="P 5"or part_lab=="P 2" 
                     continue
                     #pass
                if part_lab == "P 11":
                    chans=[48,56,57,58]
                else:
                    chans=channels
                
                print(part_lab)
                # Making Active participant list
                npa=int(part_lab[-2:])-1
                act_part.append(npa)
                act_part_lab.append(part_lab)
                
                # Removing baselines from the last participants
                
                if len(part_conds)==8:
                    pcs= part_conds[2:]
                else:
                    pcs=part_conds
                    
                
                #EMG nomalization values
                
                
                if Norm =="manual":
                    max_EMG_abs=Max_EMGs_manual[part_lab]
                elif Norm =="Abs_peak":
                    max_EMG_val_ind=np.array([EMG.max_EMG(part_conds[-i],channels=chans) for i in range(1,3)])
                
                    max_EMG_val=max_EMG_val_ind[:,0,:]
                    max_EMG_ind=max_EMG_val_ind[:,1,:]
                    max_EMG_abs=max_EMG_val.max(axis=0)
                               
                    # Printing actual maximum normalization value in the correspondent condition
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
                else:
                    # example: r;full_cicle;max
                    rphase=Norm.split(";")[-2]
                    rmet=Norm.split(";")[-1]
                    mtype=Norm.split(";")[0]
                    
                    ref_EMG=[]
                    for cond in pcs:
                        ang = max([int(s[1:]) for s in cond.mot if s.startswith(mtype)])
                        abs_val=EMG.EMG_stats(cond,rphase, Norm=[None], channels=[chans[m_n]] , metric=rmet)[:,0,:]
                        #mot_ind=cond.mot.index("r29")
                        mot_ind=cond.mot.index(mtype+str(ang))
                        ref_EMG.append(abs_val[mot_ind])
                        #mot_ind=cond.mot.index("r-29")
                        mot_ind=cond.mot.index(mtype+str(-ang))
                        ref_EMG.append(abs_val[mot_ind])
                    max_EMG_abs = np.array(ref_EMG).max()
                    max_EMG_abs = np.array([max_EMG_abs ] *4) # This is just  to get the same 4*1 format for the variable
                    # check mean emg act during approaching
                    #pick the max value
                    pass
                    
                
   
                for n_cond , cond_dat in enumerate(pcs):
                    
                    if n_cond == 2 or n_cond==3:   # Mid stiffness conditions are sipped
                        #continue
                        pass
                    if part_lab == "P 2":
                        pass
                       #print(part_lab + cond_lb[n_cond])
                       #print(n_cond)
                       #print(Norm[npa])
                    
                    # print(cond_dat.label)
                    means_stds_1p=EMG.EMG_stats(cond_dat,phase , Norm=max_EMG_abs[[m_n]] , channels=[chans[m_n]] , metric=met)[:,0,:]

                    if means_stds_1p[0].size==0:
                        print(f"{part_lab} {n_cond}")
                    
                    for nm_loc, m in enumerate(cond_dat.mot):
                        nm_glb=motions.index(m)
                        means_t[n_cond,nm_glb,npa,m_n] = means_stds_1p[nm_loc,0]
                    
                        
                        stds_t[n_cond,nm_glb,npa,m_n] = means_stds_1p[nm_loc,1]  #Not being used
                        
                        
                for condf in jcond_lb:
                    trial1=cond_lb.index(f"{condf}1")
                    trial2=cond_lb.index(f"{condf}2")
                    means_t[trial2,means_t[trial2,:,npa,m_n]==0,npa,m_n] = means_t[trial1,means_t[trial2,:,npa,m_n]==0,npa,m_n]
                    means_t[trial1,means_t[trial1,:,npa,m_n]==0,npa,m_n] = means_t[trial2,means_t[trial1,:,npa,m_n]==0,npa,m_n]
                
            if debug:
                continue
                #means=np.vstack((means,means_t[trial2,:]))
                #stds=np.vstack((stds,stds_t[trial2,:]))
                
            

            
            
            if m_n == 0:    
                exc_part=set([f"P {i+1}" for i in range(Nparts)]) ^ set(act_part_lab) 
    #Getting the mean from each pair of identical conditions
            
    means=means_t[:,:,act_part,:].reshape(Ncond,2,Nmot,len(act_part),nMusc).mean(axis = 1)
    stds=means_t[:,:,act_part,:].reshape(Ncond,2,Nmot,len(act_part),nMusc).std(axis = 1)        
 
    # Grouping muscle activities
    
    if mGroup==None:
        EMG_avg=means
        EMG_std=stds
    elif mGroup=="AntPos":
        muscles = ["SCM","SPL"]
        EMG_avg=means.reshape(Ncond,Nmot,len(act_part),int(nMusc/2),2).mean(axis = -1)
        EMG_std=stds.reshape(Ncond,Nmot,len(act_part),int(nMusc/2),2).mean(axis = -1)
       
    
    
    
    
    
     # Function to remove outliers based on IQR
    def remove_outliers(data):
        Q1 = np.percentile(data, 25, axis=-1)
        Q3 = np.percentile(data, 75, axis=-1)
        IQR = Q3 - Q1
        lower_bound = (Q1 - 1.5 * IQR).reshape(-1,1)
        upper_bound = (Q3 + 1.5 * IQR).reshape(-1,1)
        accept_cond=(data >= lower_bound).all(axis=0) & (data <= upper_bound).all(axis=0)
        exc_part=[act_part_lab[i] for i in np.arange(data.shape[-1])[np.invert( accept_cond)]]
        print("Outliers: "+" ".join(exc_part))
        clean_dat=data[:,accept_cond]
        return clean_dat 
  
    def paired_test(pGs1,pGs2,hyp=1):
        if hyp == 0:
            return scipy.stats.wilcoxon(pGs1,pGs2,axis=-1 , alternative="two-sided") 
        elif hyp == 1:
            return scipy.stats.wilcoxon(pGs1,pGs2,axis=-1 , alternative="less") 
        else:
            return scipy.stats.wilcoxon(pGs1,pGs2,axis=-1 , alternative="greater")
    
    
    for muscN,muscName in enumerate(muscles):
        print("_"*60)
        print(muscName)
        #Running statitistical tests 
        #pGroups=EMG_avg[np.arange(3)!=1,:,:,muscN]
        pGroups=EMG_avg[:,:,:,muscN]
        #print(pGroups)
       
        #print(act_part_lab)
        
        
            
        
        #pTest=scipy.stats.friedmanchisquare(list(pVals[0]),list(pVals[1]),list(pVals[2]))
        
        # Visualizing the data 
        if style==None:
            fig=plt.figure(f"EMG_{met}_{style} group: {phase} phase")
            #Getting the means mean and stds
            stds=stds[:,:,:,muscN].means(axis = -1)
            
            means=means[:,:,:,muscN].means(axis = -1)
            #stds=stds_t.reshape(2,Ncond,Nmot,Nparts).mean(axis = 0).mean(axis = -1)
            
            
            
            means=means.T.reshape(-1)
            stds=stds.T.reshape(-1)
            ##print(f"means: {means}")
            # Calculate the x positions for the bars
            x = np.array([np.linspace(0,0.5,Ncond) + i for i in range(len(motions))]).reshape(-1)
            
            ax = fig.add_subplot(len(muscles),1,muscN+1)
            ax.set_title("All parts:"  + muscName + " " + phase,fontsize = "xx-large")
            ax.set_xticks(np.arange(len(motions)),motions)
            ax.bar(x, abs(means), yerr=stds, align='center', alpha=0.7, width=0.5/Ncond, color=col*Ncond)
            if muscN==2:
                ax.legend(handles=patches,bbox_to_anchor=(1.2, 0.))
            plt.grid("on")
            ax.set_ylim([0, max(abs(means))*1.1])
            #for npv , pv in enumerate(pTest.pvalue):
                
            for mt_ind , mt in enumerate(motions):
                nmt=motions.index(mt)
                print(mt)
                pGroups_clean=remove_outliers(pGroups[:,nmt,:])
                pv1=paired_test(pGroups_clean[1], pGroups_clean[-1], 1)
                pv2=paired_test(pGroups_clean[1], pGroups_clean[-1], 1)
                barplot_annotate_brackets(mt_ind*3 , mt_ind*3+2, f"p =st: {pv1.pvalue:.2f}, dt: {pv2.pvalue:.2f}" , x , abs(means),ax)
            
        elif style=="box":
            
            
            
            #Combine this with the minimal configuration
            angs=ANG_plotmultip(parts,ref="head_rel",style="table")
            
            for mt_name , mt_key in mot_types.items():
                mots=[m for m in motions if m[0] == mt_key]
                
                if mt_key == "s":
                    side= ["Back Flexion" , "Frontal Flexion"]
                    side= ["Back High" , "Back Low" , "Frontal Low" , "Frontal High"]
                    
                elif mt_key == "l":
                    side=["Right Flexion" , "Left Flexion"]
                    side=["Right High" ,"Right Low" , "Left Low" , "Left High"]
                    
                else:
                    side=["Right Rotation" , "Left Rotation"]
                    side=["Right High" ,"Right Low" , "Left Low" , "Left High"]
                
                fig=plt.figure(f"EMG_{met}_{style} group for {mt_name}2: {phase} phase {Norm} normalization grouping {mGroup}("+",".join(exc_part)+" are excluded)")
                
                if muscN == 0:
                    fig.suptitle(f"EMG_{met}_{style} group for {mt_name}: {phase} phase("+",".join(exc_part)+" are excluded)",fontsize = "xx-large")
                    fig.tight_layout(pad=3, w_pad=0.5, h_pad=0.50) 
                    figs.append(fig)
                   
                        
                for mt_ind , mt in enumerate(mots):
                    
                    sagcond=False
                    if mt_key=="s":
                        ang=int(mt[1:])
                        if ang>0:
                            sagcond=muscN < int(len(muscles)/2)
                        else:
                            sagcond=muscN > int(len(muscles)/2)
                    
                    if mt_key=="r" or sagcond:
                        hyp=1
                        bar_col="red"
                    else:
                        hyp=2
                        bar_col="green" 
                    
                    nmt=motions.index(mt)
                    ax = fig.add_subplot(len(muscles),len(mots),(muscN*len(mots)+mt_ind+1))
                    #ax.set_title(muscName + " " + mt,fontsize = "xx-large")
                    
                    
                    
                    #Formating 
                    if muscN == 0:
                        ax.set_title(side[mt_ind],fontsize = 44)
                        ticklabs= [r"{:.0f}".format(angs[mt_key][mt_ind,ind,0])+"$^{\circ}\pm$"+"{:.0f}".format(angs[mt_key][mt_ind,ind,1])+"$^{\circ}$" for ind in np.arange(3)]
                        is_top=True
                        is_bottom=not is_top
                        fsx=30
                    elif muscN == len(muscles)-1:
                        # ticklabs= [jcond_lb[ind]+"\n"+r" ({}".format(angs[mt_key][mt_ind][ind][0])+"$^{\circ}\pm$"+"{}".format(angs[mt_key][mt_ind][ind][1])+"$^{\circ}$)"for ind in [0,-1]]
                        ticklabs= jcond_lb
                        fsx=40
                        is_top=False
                        is_bottom=not is_top
                    else:
                        ticklabs= ["",""]
                        is_top=True
                        is_bottom=True
                                         
                    
                    lableft=True
                    if muscN<int(len(muscles)/2):
                        plt.ylim(0,0.35)
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
                    else:
                        plt.ylim(0,0.8)
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

                    if mt_ind: #or mt_key != "s": Removing labels for modified image of the paer
                        lableft=False
                        #ax.set_yticks([]) # Turning off y ticks
                        
                    ax.tick_params(labeltop=is_top,labelleft=lableft,top=is_top,bottom=is_bottom,labelbottom=is_bottom,right=True)
                    plt.yticks(fontsize=30)
                    plt.xticks(fontsize=fsx)
                    plt.xlim(0.8,3.2)
                    
                    #Printing the data 
                    
                    
                    #plt.boxplot(means[:,nmt,:,muscN].T,labels=jcond_lb)
                    plt.boxplot(pGroups[:,nmt,:].T,labels=ticklabs)   
                    #heights=means[:,nmt,:].max(axis=-1)
                    heights=pGroups[:,nmt,:].max(axis=-1)
                    print(mt)
                    pairs = [[0,1],[1,2],[0,2]]
                    
                    for npair,pair in enumerate(pairs):
                        
                        pGroups_clean=remove_outliers(pGroups[pair,nmt,:])
                       
                        pv=paired_test(pGroups_clean[0], pGroups_clean[-1], hyp)
                        #pv2=paired_test(pGroups_clean[1], pGroups_clean[-1], 2)
                        if pv.pvalue < 0.05:
                                barplot_annotate_brackets(pair[0], pair[1], pv.pvalue , np.arange(1,4) , heights,ax,dh=.07*npair,col=bar_col)
                    #barplot_annotate_brackets(0, 1, f"p =st: {pv1.pvalue:.2f}, dt: {pv2.pvalue:.2f}"  , np.arange(1,3) , heights,ax)
                    
                    
         
                        
        elif style=="box_minimal":
           #Combine this with the minimal configuration
           angs=ANG_plotmultip(parts,ref="head_rel",style="table")
           cond_sset=[0,2] #excluding MS    The order affect the statistical result
           
           
           for mt_name , mt_key in mot_types.items():
               print("*"*30)
               print(mt_key)
               mots=[m for m in motions if m[0] == mt_key]
               
               if mt_key == "s":
                   side= ["Back Flexion" , "Frontal Flexion"]
                   side= ["Back Large" , "Back Small" , "Frontal Small" , "Frontal Large"]
                   
               elif mt_key == "l":
                   side=["Right Flexion" , "Left Flexion"]
                   side=["Right Large" ,"Right Small" , "Left Small" , "Left Large"]
                   
               else:
                   side=["Right Rotation" , "Left Rotation"]
                   side=["Right Large" ,"Right Small" , "Left Small" , "Left Large"]
               
               fig=plt.figure(f"EMG_{met}_{style} group for {mt_name}2: {phase} phase {Norm} normalization grouping {mGroup}("+",".join(exc_part)+" are excluded)")
               
               if muscN == 0:
                   fig.suptitle(f"EMG_{met}_{style} group for {mt_name}: {phase} phase("+",".join(exc_part)+" are excluded)",fontsize = "xx-large")
                   fig.tight_layout(pad=3, w_pad=0.5, h_pad=0.50) 
                   figs.append(fig)
                  
               ax_ord=dict()
               ax_ord.update({"r":[1,5,6,2]})        #manually placing the plots
               ax_ord.update({"s":[6,5,1,2]})
               ax_ord.update({"l":ax_ord["r"]}) 
               for mt_ind , mt in enumerate(mots):
                   
                   sagcond=False
                   if mt_key=="s":
                       ang=int(mt[1:])
                       if ang>0:
                           sagcond=muscN < int(len(muscles)/2)
                       else:
                           sagcond=muscN > int(len(muscles)/2)
                   
                   if mt_key=="r" or sagcond:
                       hyp=1
                       bar_col="red"
                   else:
                       hyp=2
                       bar_col="green" 
                   
                   nmt=motions.index(mt)
                   
                   rows=len(muscles)*2
                   cols=int(len(mots)/2)
                   ax_ind=muscN*cols + ax_ord[mt_key][mt_ind]
                   
                   #ax = fig.add_subplot(len(muscles),len(mots),(muscN*len(mots)+mt_ind+1))
                   ax = fig.add_subplot(rows,cols,ax_ind)
                   #ax.set_title(muscName + " " + mt,fontsize = "xx-large")
                   
                   #Formating 
                   
                   if muscN == 0:
                       ax.set_title(side[mt_ind],fontsize = 44)
                       ticklabs= [r"{:.0f}".format(angs[mt_key][mt_ind,ind,0])+"$^{\circ}\pm$"+"{:.0f}".format(angs[mt_key][mt_ind,ind,1])+"$^{\circ}$" for ind in cond_sset]
                       is_top=True
                       is_bottom=not is_top
                       fsx=30
                       
                   elif muscN == len(muscles)-1:
                       # ticklabs= [jcond_lb[ind]+"\n"+r" ({}".format(angs[mt_key][mt_ind][ind][0])+"$^{\circ}\pm$"+"{}".format(angs[mt_key][mt_ind][ind][1])+"$^{\circ}$)"for ind in [0,-1]]
                       ticklabs= [jcond_lb[i] for i in cond_sset]
                       fsx=40
                       is_top=False
                       is_bottom=not is_top
                   else:
                       ticklabs= [""]*len(cond_sset)
                       is_top=True
                       is_bottom=True
                                        
                   
                   
                   
                   lableft=True
                   if muscN<int(len(muscles)/2):
                       plt.ylim(0,0.35)
                       ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
                   else:
                       plt.ylim(0,0.8)
                       ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

                   if mt_ind>1: #or mt_key != "s": Removing labels for modified image of the paer
                       lableft=False
                       #ax.set_yticks([]) # Turning off y ticks
                       
                   ax.tick_params(labeltop=is_top,labelleft=lableft,top=is_top,bottom=is_bottom,labelbottom=is_bottom,right=True)
                   plt.yticks(fontsize=30)
                   plt.xticks(fontsize=fsx)
                   plt.xlim(.8,len(cond_sset)+.2)
                   
                   #Printing the data 
                   
                   
                   #plt.boxplot(means[:,nmt,:,muscN].T,labels=jcond_lb)
                   print(ticklabs)
                   plt.boxplot(pGroups[cond_sset,nmt,:].T,labels=ticklabs)   
                   print(pGroups[cond_sset,nmt,:].mean(axis=-1))
                   print(np.percentile(pGroups[cond_sset,nmt,:], 50, axis=-1))
                   
                   #heights=means[:,nmt,:].max(axis=-1)
                   heights=pGroups[cond_sset,nmt,:].max(axis=-1)
                   print(mt)
                   
                   pairs = list(combinations(range(len(cond_sset)),2))
                   
                   for npair,pair in enumerate(pairs):
                       print(pair)
                       pGind=[cond_sset[i] for i in pair]
                       print(pGind)
                       pGroups_clean=remove_outliers(pGroups[pGind,nmt,:])
                       
                       
                       # Significant diferences
                       hyp=0
                       pv=paired_test(pGroups_clean[0], pGroups_clean[1], hyp)
                       
                       if pv.pvalue < 0.05:                     
                           # Testing for significant increments
                           hyp=1
                           bar_col="red" 
                           pv=paired_test(pGroups_clean[0], pGroups_clean[1], hyp)
                           #pv2=paired_test(pGroups_clean[1], pGroups_clean[-1], 2)
                           if pv.pvalue < 0.05:
                                   barplot_annotate_brackets(pair[0], pair[1], pv.pvalue , np.arange(1,4) , heights,ax,dh=.07*npair,col=bar_col)
                           else:
                                   # Testing for significant decrements
                                   hyp=2
                                   bar_col="black"
                                   pv=paired_test(pGroups_clean[0], pGroups_clean[1], hyp)
                                   #if pv.pvalue < 0.05:
                                   #        barplot_annotate_brackets(pair[0], pair[1], pv.pvalue , np.arange(1,4) , heights,ax,dh=.07*npair,col=bar_col)
                                   barplot_annotate_brackets(pair[0], pair[1], pv.pvalue , np.arange(1,4) , heights,ax,dh=.07*npair,col=bar_col)
                        
                           
                            
                   #barplot_annotate_brackets(0, 1, f"p =st: {pv1.pvalue:.2f}, dt: {pv2.pvalue:.2f}"  , np.arange(1,3) , heights,ax)
                   
                    
                    #Stat value to discern betweeen significant increment, decrement or non-significant diference
                    # if pv1.pvalue < 0.05:
                    #     if pv2.pvalue < 0.05:
                    #         barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax,col="green")
                    #     else:
                    #         barplot_annotate_brackets(0, 1, pv1.pvalue , np.arange(1,3) , heights,ax,col="blue")
                    # else:
                    #     if pv2.pvalue < 0.05:
                    #         barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax,col="red")
                    #     else:
                    #         pass
                    #         barplot_annotate_brackets(0, 1, r"\noindent HS=LS?: {:.2f}  \\ \\  HS$>$LS?: {:.2f}".format(pv2.pvalue,pv1.pvalue), np.arange(1,4) , heights,ax,barh=0.12)
                   if save:
                        fig.tight_layout() 
                        plt.savefig(r"./Paper_plots/EMG_{}ph_for_{}.pdf".format(phase,mt_name))
                        # barplot_annotate_brackets(0, 1, f"{pv2.pvalue}" , np.arange(1,3) , heights,ax)
                        # if pv2.pvalue < 0.05:
                        #     barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax)
        elif style=="box_minimal_global":
                #Combine this with the minimal configuration
                angs=ANG_plotmultip(parts,ref="head_rel",style="table")
                cond_sset=[0,2] #excluding MS    The order affect the statistical result
                col=["white","gray"]
                patches= [Patch(facecolor=cl , label=cnd ,edgecolor="black") for cl , cnd in zip(col,["Low Stiffness" , "High Stiffness"])]
                
                for mt_name , mt_key in mot_types.items():
                    print("*"*30)
                    print(mt_key)
                    mots=[m for m in motions if m[0] == mt_key]
                
                    if mt_key == "s":
                        side= ["Back Flexion" , "Frontal Flexion"]
                        side= ["Back Large" , "Back Small" , "Frontal Small" , "Frontal Large"]
                        side= [f"Extension\n(15 deg)" , f"Flexion\n(15 deg)" ,f"Flexion\n(40 deg)"]
                        mots.pop(0)
                        
                    elif mt_key == "l":
                        side=["Right Flexion" , "Left Flexion"]
                        side=[f"Right Large\n(40 deg)" ,f"Right Small\n(15 deg)" , f"Left Small\n(15 deg)" , f"Left Large\n(40 deg)"]
                        
                    else:
                        side=["Right Rotation" , "Left Rotation"]
                        side=[f"Right Rot\n(40 deg)" ,f"Right Rot\n(15 deg)" , f"Left Rot\n(15 deg)" , f"Left Rot\n(40 deg)"]
                    
                    fig=plt.figure(f"EMG_{met}_{style} group for {mt_name}2: {phase} phase {Norm} normalization grouping {mGroup}("+",".join(exc_part)+" are excluded)")
                    mots.reverse()
                    side.reverse()
                    if muscN == 0:
                        fig.suptitle(f"EMG_{met}_{style} group for {mt_name}: {phase} phase("+",".join(exc_part)+" are excluded)",fontsize = "xx-large")
                        fig.tight_layout(pad=3, w_pad=0.5, h_pad=0.50) 
                        figs.append(fig)
                       
                    ax_ord=dict()
                    ax_ord.update({"r":[1,5,6,2]})        #manually placing the plots
                    ax_ord.update({"s":[6,5,1,2]})
                    ax_ord.update({"l":ax_ord["r"]})
                               
                    
                    if mt_key=="r":
                        hyp=1
                        bar_col="red"
                    else:
                        hyp=2
                        bar_col="green" 
                    
                    
                    rows=len(muscles)
                    cols=1
                    ax_ind=len(muscles)-muscN*cols #putting SPL first
                    
                    #ax = fig.add_subplot(len(muscles),len(mots),(muscN*len(mots)+mt_ind+1))
                    ax = fig.add_subplot(rows,cols,ax_ind)
                    #ax.set_title(muscName + " " + mt,fontsize = "xx-large")
                    
                    
                    
                    #Formating 

                        
                    
                    # ticklabs= [jcond_lb[ind]+"\n"+r" ({}".format(angs[mt_key][mt_ind][ind][0])+"$^{\circ}\pm$"+"{}".format(angs[mt_key][mt_ind][ind][1])+"$^{\circ}$)"for ind in [0,-1]]
                    ticklabs= [jcond_lb[i] for i in cond_sset]
                    fsx=30
                    is_top=False
                    is_bottom=not is_top
                    
                                         
                    
                    lableft=True
                    if muscN<int(len(muscles)/2):
                        plt.ylim(0,0.6)
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.15))
                    else:
                        plt.ylim(0,0.6)
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.15))
                    
                  
                        
                    ax.tick_params(labeltop=is_top,labelleft=lableft,top=is_top,bottom=is_bottom,labelbottom=is_bottom,right=True)
                    plt.yticks(fontsize=30)
                    plt.xticks(fontsize=fsx)
                    plt.xlim(1.2,4*2+1.6)
                    
                    
                    
                    #Printing the data 
                    nmt=[motions.index(mt) for mt in mots]
                    pGs_LS=pGroups[0,nmt]
                    pGs_HS=pGroups[2,nmt]
                    pGs_HS_LS=np.append(pGs_LS,pGs_HS,axis=1)
                    pGs_HS_LS=pGs_HS_LS.reshape(2*len(nmt),-1)

                    #plt.boxplot(means[:,nmt,:,muscN].T,labels=jcond_lb)
                    offset=2* (mt_key=="s")
                    box_pos=np.array([[2*i ,2*i+0.5] for i in range(1,1+len(mots))]).flatten()+offset
                    #plt.boxplot(pGs_HS_LS.T , positions=box_pos , labels=ticklabs*len(nmt))
                    
                    
                    #print(pGroups[cond_sset,nmt,:].mean(axis=-1))
                    #print(np.percentile(pGroups[cond_sset,nmt,:], 50, axis=-1))
                    
                    #heights=means[:,nmt,:].max(axis=-1)
                    
                    
                    
                    pairs = list(combinations(range(len(cond_sset)),2))
                    
                    for npair,pair in enumerate(pairs):
                        for nmot_in , nmot in enumerate(nmt):
                            print(pair)
                            pGind=[cond_sset[i] for i in pair]
                            print(pGind)
                            pGroups_clean=remove_outliers(pGroups[pGind,nmot,:])
                            pG_mn=pGroups_clean.mean(axis=-1)
                            pG_std=pGroups_clean.std(axis=-1)
                            ax.bar(box_pos[nmot_in*2:nmot_in*2+2], pG_mn , yerr=pG_std, align='edge', alpha=1, width=-0.5, color=col,edgecolor="black")
                            
                            # Significant diferences
                            hyp=0
                            pv=paired_test(pGroups_clean[0], pGroups_clean[1], hyp)
                            #heights=pGroups[pGind,nmot,:].max(axis=-1)
                            heights=pG_mn+pG_std
                            print(pv.pvalue)
                            if pv.pvalue < 0.05:                     
                                # Testing for significant increments
                                hyp=1
                                bar_col="red" 
                                pv=paired_test(pGroups_clean[0], pGroups_clean[1], hyp)
                                #pv2=paired_test(pGroups_clean[1], pGroups_clean[-1], 2)
                                if pv.pvalue < 0.05:
                                        barplot_annotate_brackets(pair[0], pair[1], pv.pvalue , box_pos[nmot_in*2:nmot_in*2+2] , heights,ax,dh=.07*npair+0.05,col=bar_col)
                                else:
                                        # Testing for significant decrements
                                        hyp=2
                                        bar_col="black"
                                        pv=paired_test(pGroups_clean[0], pGroups_clean[1], hyp)
                                        if pv.pvalue < 0.05:
                                                barplot_annotate_brackets(pair[0], pair[1],  pv.pvalue , box_pos[nmot_in*2:nmot_in*2+2]-0.25 , heights,ax,dh=.07*npair+0.05,col=bar_col)           
                    if muscN==0:
                        ax.set_xticks(np.arange(1,len(mots)+1)*2+offset,side)
                    else:
                        ax.set_xticks(np.arange(1,len(mots)+1)*2+offset,[""]*len(mots))
                        ax.set_title(mt_name,fontsize = 44)
                        ax.legend(handles=patches,loc="upper right",fontsize=30)
                    ax.set_ylabel(muscName,fontsize=30)
                    #barplot_annotate_brackets(0, 1, f"p =st: {pv1.pvalue:.2f}, dt: {pv2.pvalue:.2f}"  , np.arange(1,3) , heights,ax)
                    
                     
                     #Stat value to discern betweeen significant increment, decrement or non-significant diference
                     # if pv1.pvalue < 0.05:
                     #     if pv2.pvalue < 0.05:
                     #         barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax,col="green")
                     #     else:
                     #         barplot_annotate_brackets(0, 1, pv1.pvalue , np.arange(1,3) , heights,ax,col="blue")
                     # else:
                     #     if pv2.pvalue < 0.05:
                     #         barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax,col="red")
                     #     else:
                     #         pass
                     #         barplot_annotate_brackets(0, 1, r"\noindent HS=LS?: {:.2f}  \\ \\  HS$>$LS?: {:.2f}".format(pv2.pvalue,pv1.pvalue), np.arange(1,4) , heights,ax,barh=0.12)
                    if save:
                         fig.tight_layout() 
                         plt.savefig(r"./Paper_plots/EMG_{}ph_for_{}.pdf".format(phase,mt_name))
                         # barplot_annotate_brackets(0, 1, f"{pv2.pvalue}" , np.arange(1,3) , heights,ax)
                         # if pv2.pvalue < 0.05:
                         #     barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax)
        elif style=="box_minimal_3phases":
            phs=["aproaching","holding","recovery"]
            for mt_name , mt_key in mot_types.items():
                mots=[m for m in motions if m[0] == mt_key]
                print(mots)
                if mt_key == "s":
                    side= ["back" , "front"]
                else:
                    side=["right" , "left"]
                    
                
                aux= [mots[0],mots[-1]]
                mots=aux
                print(mots)
                for mt_ind , mt in enumerate(mots):
                    fig=plt.figure(f"EMG_{met}_{style} group for {mt}: 3 phases")
                    abs_ang= str(np.abs(int(mt[1:])))
                    if muscN == 0:
                        fig.suptitle(f"EMG_{met}_{style} group for {mt_name},{side[mt_ind]} {abs_ang} : 3 phases("+",".join(exc_part)+" are excluded)",fontsize = "xx-large")
                        fig.tight_layout(pad=3, w_pad=0.5, h_pad=0.50) 
                        figs.append(fig)
                    
                    nmt=motions.index(mt)
                    print(phs)
                    ax = fig.add_subplot(len(muscles),3,(muscN*3+phs.index(phase)+1))
                    
                    
                    
                    ax.set_title(muscle + " " + phase ,fontsize = 22)
                    
                    plt.boxplot(means[[0,-1],nmt,:,muscN].T,labels=[jcond_lb[0] , jcond_lb[-1]])
                    plt.yticks(fontsize=20)
                    plt.xticks(fontsize=20)
                    heights=means[[0,-1],nmt,:,muscN].max(axis=-1)
                    print(mt)
                    pGroups_clean=remove_outliers(pGroups[:,nmt,:])
                    pv1=paired_test(pGroups_clean[1], pGroups_clean[-1], 1)
                    pv2=paired_test(pGroups_clean[1], pGroups_clean[-1], 1)
                    
                    if pv1.pvalue < 0.05:
                        if pv2.pvalue < 0.05:
                            barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax,col="green")
                        else:
                            barplot_annotate_brackets(0, 1, pv1.pvalue , np.arange(1,3) , heights,ax,col="blue")
                    else:
                        if pv2.pvalue < 0.05:
                            barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax,col="red")
                        else:
                            barplot_annotate_brackets(0, 1, r"\noindent HS=LS?: {:.2f}  \\ \\  HS$>$LS?: {:.2f}".format(pv2.pvalue,pv1.pvalue), np.arange(1,4) , heights,ax,barh=0.12)
                    if save:
                        fig.tight_layout() 
                        plt.savefig(r"./Paper_plots/EMG_3phs_for_{}.pdf".format(mt))
                        
                    # barplot_annotate_brackets(0, 1, f"{pv2.pvalue}" , np.arange(1,3) , heights,ax)
                    # if pv2.pvalue < 0.05:
                    #     barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax)
        elif style=="table":
            stds=stds.mean[:,:,:,muscN](axis = -1)
            means=means.mean[:,:,:,muscN](axis = -1)
            
            heads=[muscle]
            
            for mot in ["Fflex/Rbend" , "Bflex/LBend"]:
                heads+=[f"{mot}_LS"]+ [f"{mot}_HS"] + ["HS != LS?"] + ["HS > LS?"]
            
            EMG_table.append(heads)
            for mt_name , mt_key in mot_types.items():
                EMG_table.append([mt_name])
                
                mots=[m for m in motions if m[0] == mt_key]
                if mt_key == "s":
                    side= ["back" , "front"]
                else:
                    side=["right" , "left"]
                    
                aux= [mots[0],mots[-1]]
                mots=aux
                
                for mt_ind , mt in enumerate(mots):
    
                    nmt=motions.index(mt)
                    print(mt)
    
                    abs_ang= [side[mt_ind]] + [str(np.abs(int(mt[1:])))]
                    
                    
                    pGroups_clean=remove_outliers(pGroups[:,nmt,:])
                    pv1=paired_test(pGroups_clean[1], pGroups_clean[-1], 1)
                    pv2=paired_test(pGroups_clean[1], pGroups_clean[-1], 1)
                    
                    means[0,nmt]
                    [EMG_table[-1].append("{:.2f} +-{:.2f}".format(means[cnd,nmt],stds[cnd,nmt]) ) for cnd in [0,2]]
                    EMG_table[-1].append(pv2.pvalue)
                    EMG_table[-1].append(pv1.pvalue)
            if muscN==3:
                print(tabulate.tabulate(EMG_table, headers = heads, tablefmt="fancy_grid"))  
                with open(r"G:/My Drive/PhD/My research/Experiments/EMG and MOCAP measurements/5- 6 bar rigid blockage, 6 users/Plots/table2.csv","w") as f:
                    writee=csv.writer(f)
                    writee.writerows(EMG_table)
                            
                                
                        
                    
                            
                            
                
        elif style=="lines":
                    for mt_name , mt_key in mot_types.items():
                        mots=[m for m in motions if m[0] == mt_key]
                        
                        fig=plt.figure(f"EMG_{met}_{style} group for {mt_name}: {phase} phase")
                        if muscN == 0:
                            fig.suptitle(f"EMG_{met}_{style} group for {mt_name}: {phase} phase {Norm} normalization ("+",".join(exc_part)+" are excluded)",fontsize = "xx-large")
                            fig.tight_layout(pad=3, w_pad=0.5, h_pad=0.50) 
                            figs.append(fig)
                            
                            
                        for mt_ind , mt in enumerate(mots):
                            nmt=motions.index(mt)
                            ax = fig.add_subplot(len(muscles),len(mots),(muscN*len(mots)+mt_ind+1))
                            ax.set_title(muscle + " " + mt,fontsize = "xx-large")
                            ax.set_xticks(np.arange(3),jcond_lb)
                            
                            legs=[f"P{ap+1}" for ap in act_part]
                            ax.plot(means[:,nmt,:,muscN],label=legs)
                            if muscN == 0 and nmt==0:
                                ax.legend()
                                
        elif style=="lines_minimal":
                    for mt_name , mt_key in mot_types.items():
                        mots=[m for m in motions if m[0] == mt_key]
                        
                        #fig=plt.figure(f"EMG_{met}_{style} group for {mt_name}: {phase} phase {Norm} normalization")
                        fig=plt.figure(f"EMG_{met}_{style} group for {mt_name}2: {phase} phase {Norm} normalization grouping {mGroup}("+",".join(exc_part)+" are excluded)")
                        
                        if muscN == 0:
                            fig.suptitle(f"EMG_{met}_{style} group for {mt_name}: {phase} phase ("+",".join(exc_part)+" are excluded)",fontsize = "xx-large")
                            fig.tight_layout(pad=3, w_pad=0.5, h_pad=0.50) 
                            figs.append(fig)
                            
                        aux= [mots[0],mots[-1]]
                        mots=aux    
                        rng=np.arange(3)!=1
                        for mt_ind , mt in enumerate(mots):
                            nmt=motions.index(mt)
                            ax = fig.add_subplot(len(muscles),len(mots),(muscN*len(mots)+mt_ind+1))
                            ax.set_title(muscle + " " + mt,fontsize = "xx-large")
                            ax.set_xticks(np.arange(3)[rng],[jcond_lb[0],jcond_lb[2]])
                            
                            legs=[f"P{ap+1}" for ap in act_part]
                            
                            ax.plot(means[rng,nmt,:,muscN],label=legs)
                            if muscN == 0 and mt_ind==0:
                                ax.legend(loc="upper right")
   
            
            
            
           # ax.update(hspace=0.5)
    
    #mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()  
         
    # print("Means matrix")
    # print(means_t[:,:,act_part])
    if save:
                plt.savefig(r"./Plots/9302023/EMG_{}_{} group {} phase.pdf".format(met,style,phase))


def EMG_plotmultip_EMBC(parts,phase="holding",save=False,debug=False, Norm="manual", channels=[0,1,2,3], met="mean" ,style=None , mGroup=None):
    muscles = ['(R) Sternocleidomastoid', '(L) Sternocleidomastoid','(R) Para spinal','(L) Para spinal']
    muscles = ['right STR', 'left STR','right SPL','left SPL']
    Nparts= 7
    
    motions = parts["P 1"][0].mot
    motions.sort(key=crit)
    Nmot= len(motions)
    nMusc= len(muscles)
    cond_lb = ['LS1', 'LS2', 'MS1', 'MS2', 'HS1', 'HS2']
    jcond_lb= ['F','LS', 'MS', 'HS']
    
    stds=dict() # Container with stds EMG values of each INVOLVED participant

    print("Note: intermediate stiffness is not being processed")
    
    print(motions)
    custNorm={}
    
    exc_part=[]
    
    
    
    
    mot_types={"axial rotation":"r" , "sagital flexion":"s"}
    figs=[] #Figure container
    EMG_table=[] #Table container
    
    col_idx = np.linspace(0,255,len(jcond_lb)).astype(int)
    col = list(mpl.cm.gist_rainbow(col_idx))
    patches= [Patch(color=cl , label=cnd) for cl , cnd in zip(col,jcond_lb)]
    
    # Pick the statistics values for each condition
    Ncond=len(jcond_lb)  
    means=np.zeros((Ncond,Nmot,Nparts,nMusc))
    stds=np.zeros((Ncond,Nmot,Nparts,nMusc))
    
    
    for m_n , muscle in enumerate(muscles):
            print(muscle)
        
            act_part=[] #list with the active participants
            act_part_lab=[] #label list with the active participants
            
            for  part_lab, part in parts.items():
   
              #Filtering participant
                if part_lab=="P 1":# or part_lab=="P 7" or part_lab=="P 2" or part_lab=="P 5"or part_lab=="P 2" 
                     #continue
                     pass
                    
                # Making Active participant list
                npa=int(part_lab[-2:])-1
                act_part.append(npa)
                act_part_lab.append(part_lab)               
                          
                
                #Getting the participant normaliztion value 
                max_EMG_abs = EMG.norm_EMG_part(part_lab,part,channels,m_n,nMusc,crit=Norm) 

   
                for n_cond , cond_dat in enumerate(part):
                    
                    if n_cond == 2 or n_cond==3:   # Mid stiffness conditions are sipped
                        #continue
                        pass
                    if part_lab == "P 2":
                        pass
                       #print(part_lab + cond_lb[n_cond])
                       #print(n_cond)
                       #print(Norm[npa])
                    
                    # print(cond_dat.label)
                    means_stds_1p=EMG.EMG_stats(cond_dat,phase , Norm=max_EMG_abs[[m_n]] , channels=[channels[m_n]] , metric=met)[:,0,:]

                    if means_stds_1p[0].size==0:
                        print(f"{part_lab} {n_cond}")
                    
                    for nm_loc, m in enumerate(cond_dat.mot):
                        nm_glb=motions.index(m)
                        means[n_cond,nm_glb,npa,m_n] = means_stds_1p[nm_loc,0]
                    
                        
                        stds[n_cond,nm_glb,npa,m_n] = means_stds_1p[nm_loc,1]  #Not being used
                        
                        
            print("Normalization")
            print(part_lab)
            print(max_EMG_abs)
                
            if debug:
                continue


            
            
            if m_n == 0:    
                exc_part=set([f"P {i+1}" for i in range(Nparts)]) ^ set(act_part_lab) 
    #Getting the mean from each pair of identical conditions
    means=means[:,:,act_part,:]
    stds=stds[:,:,act_part,:]
    # Grouping muscle activities
    
    if mGroup==None:
        EMG_avg=means
        EMG_std=stds
    elif mGroup=="AntPos":
        muscles = ["SCM","SPL"]
        print(means.shape)
        EMG_avg=means.reshape(Ncond,Nmot,len(act_part),int(nMusc/2),2).mean(axis = -1)
        EMG_std=stds.reshape(Ncond,Nmot,len(act_part),int(nMusc/2),2).mean(axis = -1)
    elif mGroup=="ContIpsi":
        muscles = ["conSCM","ipsiSCM","conSPL","ipsiSPL"]
        
        EMG_avg=means
        EMG_std=stds
        r15_ind=motions.index("r15")
        rm15_ind=motions.index("r-15")
        s15_ind=motions.index("s15")
        sm15_ind=motions.index("s-15")
        EMG_avg[:,r15_ind]=EMG_avg[:,r15_ind][:,:,[0,1,2,3]]
        EMG_avg[:,rm15_ind]=EMG_avg[:,rm15_ind][:,:,[1,0,3,2]]
        EMG_std[:,r15_ind]=EMG_std[:,r15_ind][:,:,[0,1,2,3]]
        EMG_std[:,rm15_ind]=EMG_std[:,rm15_ind][:,:,[1,0,3,2]]
        print(EMG_avg.shape)
        print("before")
        print(EMG_avg[:,:2,:,0])
        EMG_avg=EMG_avg.reshape(Ncond,int(Nmot/2),len(act_part)*2,4)
        print("after")
        print(EMG_avg[:,0,:,0])
        EMG_std=EMG_avg.reshape(Ncond,int(Nmot/2),len(act_part)*2,4)
    
    
    
    
    
     # Function to remove outliers based on IQR
    def remove_outliers(data):
        Q1 = np.percentile(data, 25, axis=-1)
        Q3 = np.percentile(data, 75, axis=-1)
        IQR = Q3 - Q1
        lower_bound = (Q1 - 1.5 * IQR).reshape(-1,1)
        upper_bound = (Q3 + 1.5 * IQR).reshape(-1,1)
        accept_cond=(data >= lower_bound).all(axis=0) & (data <= upper_bound).all(axis=0)
        exc_part=[act_part_lab[i] for i in np.arange(data.shape[-1])[np.invert( accept_cond)]]
        print("Outliers: "+" ".join(exc_part))
        clean_dat=data[:,accept_cond]
        return clean_dat 
  
    def paired_test(pGs1,pGs2,hyp=1):
        if hyp == 0:
            return scipy.stats.wilcoxon(pGs1,pGs2,axis=-1 , alternative="two-sided") 
        elif hyp == 1:
            return scipy.stats.wilcoxon(pGs1,pGs2,axis=-1 , alternative="less") 
        else:
            return scipy.stats.wilcoxon(pGs1,pGs2,axis=-1 , alternative="greater")
    
    
    for muscN,muscName in enumerate(muscles):
        print("_"*60)
        print(muscName)
        #Running statitistical tests 
        #pGroups=EMG_avg[np.arange(3)!=1,:,:,muscN]
        pGroups=EMG_avg[:,:,:,muscN]
        #print(pGroups)
       
        #print(act_part_lab)
        
        
            
        
        #pTest=scipy.stats.friedmanchisquare(list(pVals[0]),list(pVals[1]),list(pVals[2]))
        
        # Visualizing the data 
        if style==None:
            fig=plt.figure(f"EMG_{met}_{style} group: {phase} phase")
            #Getting the means mean and stds
            stds=stds[:,:,:,muscN].means(axis = -1)
            
            means=means[:,:,:,muscN].means(axis = -1)
            #stds=stds_t.reshape(2,Ncond,Nmot,Nparts).mean(axis = 0).mean(axis = -1)
            
            
            
            means=means.T.reshape(-1)
            stds=stds.T.reshape(-1)
            ##print(f"means: {means}")
            # Calculate the x positions for the bars
            x = np.array([np.linspace(0,0.5,Ncond) + i for i in range(len(motions))]).reshape(-1)
            
            ax = fig.add_subplot(len(muscles),1,muscN+1)
            ax.set_title("All parts:"  + muscName + " " + phase,fontsize = "xx-large")
            ax.set_xticks(np.arange(len(motions)),motions)
            ax.bar(x, abs(means), yerr=stds, align='center', alpha=0.7, width=0.5/Ncond, color=col*Ncond)
            if muscN==2:
                ax.legend(handles=patches,bbox_to_anchor=(1.2, 0.))
            plt.grid("on")
            ax.set_ylim([0, max(abs(means))*1.1])
            #for npv , pv in enumerate(pTest.pvalue):
                
            for mt_ind , mt in enumerate(motions):
                nmt=motions.index(mt)
                print(mt)
                pGroups_clean=remove_outliers(pGroups[:,nmt,:])
                pv1=paired_test(pGroups_clean[1], pGroups_clean[-1], 1)
                pv2=paired_test(pGroups_clean[1], pGroups_clean[-1], 1)
                barplot_annotate_brackets(mt_ind*3 , mt_ind*3+2, f"p =st: {pv1.pvalue:.2f}, dt: {pv2.pvalue:.2f}" , x , abs(means),ax)
            
        elif style=="box":
            
            
            
            #Combine this with the minimal configuration
            angs=ANG_plotmultip(parts,ref="head_rel",style="table")
            
            for mt_name , mt_key in mot_types.items():
                mots=[m for m in motions if m[0] == mt_key]
                
                if mt_key == "s":
                    side= ["Back Flexion" , "Frontal Flexion"]
                    side= ["Back High" , "Back Low" , "Frontal Low" , "Frontal High"]
                    
                elif mt_key == "l":
                    side=["Right Flexion" , "Left Flexion"]
                    side=["Right High" ,"Right Low" , "Left Low" , "Left High"]
                    
                else:
                    side=["Right Rotation" , "Left Rotation"]
                    side=["Right High" ,"Right Low" , "Left Low" , "Left High"]
                
                fig=plt.figure(f"EMG_{met}_{style} group for {mt_name}2: {phase} phase {Norm} normalization grouping {mGroup}("+",".join(exc_part)+" are excluded)")
                
                if muscN == 0:
                    fig.suptitle(f"EMG_{met}_{style} group for {mt_name}: {phase} phase("+",".join(exc_part)+" are excluded)",fontsize = "xx-large")
                    fig.tight_layout(pad=3, w_pad=0.5, h_pad=0.50) 
                    figs.append(fig)
                   
                        
                for mt_ind , mt in enumerate(mots):
                    
                    sagcond=False
                    if mt_key=="s":
                        ang=int(mt[1:])
                        if ang>0:
                            sagcond=muscN < int(len(muscles)/2)
                        else:
                            sagcond=muscN > int(len(muscles)/2)
                    
                    if mt_key=="r" or sagcond:
                        hyp=1
                        bar_col="red"
                    else:
                        hyp=2
                        bar_col="green" 
                    
                    nmt=motions.index(mt)
                    ax = fig.add_subplot(len(muscles),len(mots),(muscN*len(mots)+mt_ind+1))
                    #ax.set_title(muscName + " " + mt,fontsize = "xx-large")
                    
                    
                    
                    #Formating 
                    if muscN == 0:
                        ax.set_title(side[mt_ind],fontsize = 44)
                        ticklabs= [r"{:.0f}".format(angs[mt_key][mt_ind,ind,0])+"$^{\circ}\pm$"+"{:.0f}".format(angs[mt_key][mt_ind,ind,1])+"$^{\circ}$" for ind in np.arange(3)]
                        is_top=True
                        is_bottom=not is_top
                        fsx=30
                    elif muscN == len(muscles)-1:
                        # ticklabs= [jcond_lb[ind]+"\n"+r" ({}".format(angs[mt_key][mt_ind][ind][0])+"$^{\circ}\pm$"+"{}".format(angs[mt_key][mt_ind][ind][1])+"$^{\circ}$)"for ind in [0,-1]]
                        ticklabs= jcond_lb
                        fsx=40
                        is_top=False
                        is_bottom=not is_top
                    else:
                        ticklabs= ["",""]
                        is_top=True
                        is_bottom=True
                                         
                    
                    lableft=True
                    if muscN<int(len(muscles)/2):
                        plt.ylim(0,0.35)
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
                    else:
                        plt.ylim(0,0.8)
                        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

                    if mt_ind: #or mt_key != "s": Removing labels for modified image of the paer
                        lableft=False
                        #ax.set_yticks([]) # Turning off y ticks
                        
                    ax.tick_params(labeltop=is_top,labelleft=lableft,top=is_top,bottom=is_bottom,labelbottom=is_bottom,right=True)
                    plt.yticks(fontsize=30)
                    plt.xticks(fontsize=fsx)
                    plt.xlim(0.8,3.2)
                    
                    #Printing the data 
                    
                    
                    #plt.boxplot(means[:,nmt,:,muscN].T,labels=jcond_lb)
                    plt.boxplot(pGroups[:,nmt,:].T,labels=ticklabs)   
                    #heights=means[:,nmt,:].max(axis=-1)
                    heights=pGroups[:,nmt,:].max(axis=-1)
                    print(mt)
                    pairs = [[0,1],[1,2],[0,2]]
                    
                    for npair,pair in enumerate(pairs):
                        
                        pGroups_clean=remove_outliers(pGroups[pair,nmt,:])
                       
                        pv=paired_test(pGroups_clean[0], pGroups_clean[-1], hyp)
                        #pv2=paired_test(pGroups_clean[1], pGroups_clean[-1], 2)
                        if pv.pvalue < 0.05:
                                barplot_annotate_brackets(pair[0], pair[1], pv.pvalue , np.arange(1,4) , heights,ax,dh=.07*npair,col=bar_col)
                    #barplot_annotate_brackets(0, 1, f"p =st: {pv1.pvalue:.2f}, dt: {pv2.pvalue:.2f}"  , np.arange(1,3) , heights,ax)
                    
                    
         
                        
        elif style=="box_minimal":
           #Combine this with the minimal configuration
           angs=ANG_plotmultip(parts,ref="head_rel",style="table")
           cond_sset=[0,2] #excluding MS    The order affect the statistical result
           
           
           for mt_name , mt_key in mot_types.items():
               print("*"*30)
               print(mt_key)
               mots=[m for m in motions if m[0] == mt_key]
               
               if mt_key == "s":
                   side= ["Back Flexion" , "Frontal Flexion"]
                   side= ["Back Large" , "Back Small" , "Frontal Small" , "Frontal Large"]
                   
               elif mt_key == "l":
                   side=["Right Flexion" , "Left Flexion"]
                   side=["Right Large" ,"Right Small" , "Left Small" , "Left Large"]
                   
               else:
                   side=["Right Rotation" , "Left Rotation"]
                   side=["Right Large" ,"Right Small" , "Left Small" , "Left Large"]
               
               fig=plt.figure(f"EMG_{met}_{style} group for {mt_name}2: {phase} phase {Norm} normalization grouping {mGroup}("+",".join(exc_part)+" are excluded)")
               
               if muscN == 0:
                   fig.suptitle(f"EMG_{met}_{style} group for {mt_name}: {phase} phase("+",".join(exc_part)+" are excluded)",fontsize = "xx-large")
                   fig.tight_layout(pad=3, w_pad=0.5, h_pad=0.50) 
                   figs.append(fig)
                  
               ax_ord=dict()
               ax_ord.update({"r":[1,5,6,2]})        #manually placing the plots
               ax_ord.update({"s":[6,5,1,2]})
               ax_ord.update({"l":ax_ord["r"]}) 
               for mt_ind , mt in enumerate(mots):
                   
                   sagcond=False
                   if mt_key=="s":
                       ang=int(mt[1:])
                       if ang>0:
                           sagcond=muscN < int(len(muscles)/2)
                       else:
                           sagcond=muscN > int(len(muscles)/2)
                   
                   if mt_key=="r" or sagcond:
                       hyp=1
                       bar_col="red"
                   else:
                       hyp=2
                       bar_col="green" 
                   
                   nmt=motions.index(mt)
                   
                   rows=len(muscles)*2
                   cols=int(len(mots)/2)
                   ax_ind=muscN*cols + ax_ord[mt_key][mt_ind]
                   
                   #ax = fig.add_subplot(len(muscles),len(mots),(muscN*len(mots)+mt_ind+1))
                   ax = fig.add_subplot(rows,cols,ax_ind)
                   #ax.set_title(muscName + " " + mt,fontsize = "xx-large")
                   
                   #Formating 
                   
                   if muscN == 0:
                       ax.set_title(side[mt_ind],fontsize = 44)
                       ticklabs= [r"{:.0f}".format(angs[mt_key][mt_ind,ind,0])+"$^{\circ}\pm$"+"{:.0f}".format(angs[mt_key][mt_ind,ind,1])+"$^{\circ}$" for ind in cond_sset]
                       is_top=True
                       is_bottom=not is_top
                       fsx=30
                       
                   elif muscN == len(muscles)-1:
                       # ticklabs= [jcond_lb[ind]+"\n"+r" ({}".format(angs[mt_key][mt_ind][ind][0])+"$^{\circ}\pm$"+"{}".format(angs[mt_key][mt_ind][ind][1])+"$^{\circ}$)"for ind in [0,-1]]
                       ticklabs= [jcond_lb[i] for i in cond_sset]
                       fsx=40
                       is_top=False
                       is_bottom=not is_top
                   else:
                       ticklabs= [""]*len(cond_sset)
                       is_top=True
                       is_bottom=True
                                        
                   
                   
                   
                   lableft=True
                   if muscN<int(len(muscles)/2):
                       plt.ylim(0,0.35)
                       ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
                   else:
                       plt.ylim(0,0.8)
                       ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

                   if mt_ind>1: #or mt_key != "s": Removing labels for modified image of the paer
                       lableft=False
                       #ax.set_yticks([]) # Turning off y ticks
                       
                   ax.tick_params(labeltop=is_top,labelleft=lableft,top=is_top,bottom=is_bottom,labelbottom=is_bottom,right=True)
                   plt.yticks(fontsize=30)
                   plt.xticks(fontsize=fsx)
                   plt.xlim(.8,len(cond_sset)+.2)
                   
                   #Printing the data 
                   
                   
                   #plt.boxplot(means[:,nmt,:,muscN].T,labels=jcond_lb)
                   print(ticklabs)
                   plt.boxplot(pGroups[cond_sset,nmt,:].T,labels=ticklabs)   
                   print(pGroups[cond_sset,nmt,:].mean(axis=-1))
                   print(np.percentile(pGroups[cond_sset,nmt,:], 50, axis=-1))
                   
                   #heights=means[:,nmt,:].max(axis=-1)
                   heights=pGroups[cond_sset,nmt,:].max(axis=-1)
                   print(mt)
                   
                   pairs = list(combinations(range(len(cond_sset)),2))
                   
                   for npair,pair in enumerate(pairs):
                       print(pair)
                       pGind=[cond_sset[i] for i in pair]
                       print(pGind)
                       pGroups_clean=remove_outliers(pGroups[pGind,nmt,:])
                       
                       
                       # Significant diferences
                       hyp=0
                       pv=paired_test(pGroups_clean[0], pGroups_clean[1], hyp)
                       
                       if pv.pvalue < 0.05:                     
                           # Testing for significant increments
                           hyp=1
                           bar_col="red" 
                           pv=paired_test(pGroups_clean[0], pGroups_clean[1], hyp)
                           #pv2=paired_test(pGroups_clean[1], pGroups_clean[-1], 2)
                           if pv.pvalue < 0.05:
                                   barplot_annotate_brackets(pair[0], pair[1], pv.pvalue , np.arange(1,4) , heights,ax,dh=.07*npair,col=bar_col)
                           else:
                                   # Testing for significant decrements
                                   hyp=2
                                   bar_col="black"
                                   pv=paired_test(pGroups_clean[0], pGroups_clean[1], hyp)
                                   #if pv.pvalue < 0.05:
                                   #        barplot_annotate_brackets(pair[0], pair[1], pv.pvalue , np.arange(1,4) , heights,ax,dh=.07*npair,col=bar_col)
                                   barplot_annotate_brackets(pair[0], pair[1], pv.pvalue , np.arange(1,4) , heights,ax,dh=.07*npair,col=bar_col)
                        
                           
                            
                   #barplot_annotate_brackets(0, 1, f"p =st: {pv1.pvalue:.2f}, dt: {pv2.pvalue:.2f}"  , np.arange(1,3) , heights,ax)
                   
                    
                    #Stat value to discern betweeen significant increment, decrement or non-significant diference
                    # if pv1.pvalue < 0.05:
                    #     if pv2.pvalue < 0.05:
                    #         barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax,col="green")
                    #     else:
                    #         barplot_annotate_brackets(0, 1, pv1.pvalue , np.arange(1,3) , heights,ax,col="blue")
                    # else:
                    #     if pv2.pvalue < 0.05:
                    #         barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax,col="red")
                    #     else:
                    #         pass
                    #         barplot_annotate_brackets(0, 1, r"\noindent HS=LS?: {:.2f}  \\ \\  HS$>$LS?: {:.2f}".format(pv2.pvalue,pv1.pvalue), np.arange(1,4) , heights,ax,barh=0.12)
                   if save:
                        fig.tight_layout() 
                        plt.savefig(r"./Paper_plots/EMG_{}ph_for_{}.pdf".format(phase,mt_name))
                        # barplot_annotate_brackets(0, 1, f"{pv2.pvalue}" , np.arange(1,3) , heights,ax)
                        # if pv2.pvalue < 0.05:
                        #     barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax)
        elif style=="box_minimal_global":
                #Combine this with the minimal configuration
                #angs=ANG_plotmultip(parts,ref="head_rel",style="table")
                cond_sset=[0,1,3] #excluding MS    The order affect the statistical result
                col=["white","lightgray","darkgray","dimgray"]
                patches= [Patch(facecolor=cl , label=cnd ,edgecolor="black") for cl , cnd in zip(col,["Baseline","Low Stiffness","Middle Stiffness","High Stiffness"])]
                
                for mt_name , mt_key in mot_types.items():
                    print("*"*30)
                    print(mt_key)
                    mots=[m for m in motions if m[0] == mt_key]
                    
                    if mt_key == "s":
                        side= ["Back Flexion" , "Frontal Flexion"]
                        side= ["Back Large" , "Back Small" , "Frontal Small" , "Frontal Large"]
                        side= [f"Extension\n(15 deg)" , f"Flexion\n(15 deg)"]
                        #mots.pop(0)
                        
                    elif mt_key == "l":
                        side=["Right Flexion" , "Left Flexion"]
                        side=[f"Right Large\n(40 deg)" ,f"Right Small\n(15 deg)" , f"Left Small\n(15 deg)" , f"Left Large\n(40 deg)"]
                        continue
                    else:
                        side=["Right Rotation" , "Left Rotation"]
                        side=[f"Right Rot\n(15 deg)" , f"Left Rot\n(15 deg)"]
                        continue
                    
                    fig=plt.figure(f"EMG_{met}_{style} group for {mt_name}2: {phase} phase {Norm} normalization grouping {mGroup}("+",".join(exc_part)+" are excluded)")
                    mots.reverse()
                    
                    side.reverse()
                    if muscN == 0:
                        fig.suptitle(f"EMG_{met}_{style} group for {mt_name}: {phase} phase("+",".join(exc_part)+" are excluded)",fontsize = "xx-large")
                        #fig.tight_layout(pad=3, w_pad=0.5, h_pad=0.50) 
                        figs.append(fig)
                        
                        #Formating 
                        # ticklabs= [jcond_lb[ind]+"\n"+r" ({}".format(angs[mt_key][mt_ind][ind][0])+"$^{\circ}\pm$"+"{}".format(angs[mt_key][mt_ind][ind][1])+"$^{\circ}$)"for ind in [0,-1]]
                        ticklabs= [jcond_lb[i] for i in cond_sset]
                        fsx=30
                        is_top=False
                        is_bottom=not is_top
                        lableft=True
                        
                        
                        
                        rows=1
                        cols=2
                        axf = fig.add_subplot(rows,cols,1)
                        axf.tick_params(labeltop=is_top,labelleft=lableft,top=is_top,bottom=is_bottom,labelbottom=is_bottom,right=True)
                        plt.yticks(fontsize=30)
                        plt.xticks(fontsize=fsx)
                        plt.xlim(2.2,8.3)
                        plt.ylim(-0.1,0.4)

                        
                        lableft=False
                        axe = fig.add_subplot(rows,cols,2)
                        axe.tick_params(labeltop=is_top,labelleft=lableft,top=is_top,bottom=is_bottom,labelbottom=is_bottom,right=True)
                        plt.yticks(fontsize=30)
                        plt.xticks(fontsize=fsx)
                        plt.ylim(-0.1,0.4)
                        
                        
                        
                        
                        
                    
                    
                    #ax = fig.add_subplot(len(muscles),len(mots),(muscN*len(mots)+mt_ind+1))
                    
                    #ax.set_title(muscName + " " + mt,fontsize = "xx-large")
                                    
                    
                    
                                         
                    
                    lableft=True
                    if muscN<int(len(muscles)/2):
                        pass
                        #plt.ylim(0,0.6)
                        #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.15))
                    else:
                        pass
                        #plt.ylim(0,0.6)
                        #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.15))
                    
                  
                        
                    
                    
                    
                    
                    #Printing the data 
                    nmt=[motions.index(mt) for mt in mots]
                    pGs_LS=pGroups[0,nmt]
                    pGs_HS=pGroups[2,nmt]
                    pGs_HS_LS=np.append(pGs_LS,pGs_HS,axis=1)
                    pGs_HS_LS=pGs_HS_LS.reshape(2*len(nmt),-1)

                    #plt.boxplot(means[:,nmt,:,muscN].T,labels=jcond_lb)
                    offset=0* (mt_key=="s")
                    box_pos=np.array([[3*i ,3*i+0.5,3*i+1,3*i+1.5] for i in range(1,1+len(mots))])+offset
                    box_pos=np.array([[3*i ,3*i+0.5,3*i+1] for i in range(1,1+len(muscles))])+offset
                    #plt.boxplot(pGs_HS_LS.T , positions=box_pos , labels=ticklabs*len(nmt))
                    
                    
                    #print(pGroups[cond_sset,nmt,:].mean(axis=-1))
                    #print(np.percentile(pGroups[cond_sset,nmt,:], 50, axis=-1))
                    
                    #heights=means[:,nmt,:].max(axis=-1)
                    pG_mn=pGroups.mean(axis=-1)
                    pG_std=pGroups.std(axis=-1)
                    
                    for nmot_in , nmot in enumerate(nmt):
                        ax=fig.axes[nmot_in]
                        ax.bar(box_pos[muscN], pG_mn[cond_sset,nmot] , yerr=pG_std[cond_sset,nmot_in] , align='edge', alpha=1, width=-0.5, color=[col[i] for i in cond_sset],edgecolor="black")
                       
                        
                        pairs = [[0,1],[1,2],[2,3]]
                        pairs = [[0,1],[1,3],[0,3]]
                        for npair,pair in enumerate(pairs):
                                p1=pair[0]
                                p2=pair[1]
                                p1_in=cond_sset.index(p1)
                                p2_in=cond_sset.index(p2)
                                print(pair)
                              
                                # Significant diferences
                                hyp=0
                                pv=paired_test(pGroups[p1,nmot], pGroups[p2,nmot], hyp)
                                #heights=pGroups[pGind,nmot,:].max(axis=-1)
                                heights=pG_mn+pG_std
                                print(heights.shape)
                                print(pv.pvalue)
                                if pv.pvalue < 0.05:                     
                                    # Testing for significant increments
                                    hyp=1
                                    bar_col="black" 
                                    pv=paired_test(pGroups[p1,nmot], pGroups[p2,nmot], hyp)
                                    #pv2=paired_test(pGroups_clean[1], pGroups_clean[-1], 2)
                                    print()
                                    if pv.pvalue < 0.05:
                                            barplot_annotate_brackets(p1_in,p2_in, pv.pvalue , box_pos[muscN]-0.25 , heights[cond_sset,nmot],ax,dh=.07*npair+0.05,col=bar_col)
                                    else:
                                            # Testing for significant decrements
                                            hyp=2
                                            bar_col="green"
                                            pv=paired_test(pGroups[p1,nmot], pGroups[p2,nmot], hyp)
                                            if pv.pvalue < 0.05:
                                                    barplot_annotate_brackets(p1_in,p2_in , pv.pvalue , box_pos[muscN]-0.25 , heights[:,nmot],ax,dh=.07*npair+0.05,col=bar_col)           
                    if muscN==0:
                        print(np.arange(1,len(mots)+1)*2)
                        axe.set_xticks(np.arange(1,len(muscles)+1)*3+0.5,["conSTR","ipsSPL"])
                        axf.set_xticks(np.arange(1,len(muscles)+1)*3+0.5,["ipsSTR","conSPL"])
                        axf.legend(handles=[patches[i] for i in cond_sset],loc="upper right",fontsize=30)
                        axf.set_title("Flexion (15 deg)",fontsize = 44)
                        axe.set_title("Extension (15 deg)",fontsize = 44)
                        axf.set_ylabel("normalized EMG",fontsize=30)
                    #barplot_annotate_brackets(0, 1, f"p =st: {pv1.pvalue:.2f}, dt: {pv2.pvalue:.2f}"  , np.arange(1,3) , heights,ax)
                    
                     
                     #Stat value to discern betweeen significant increment, decrement or non-significant diference
                     # if pv1.pvalue < 0.05:
                     #     if pv2.pvalue < 0.05:
                     #         barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax,col="green")
                     #     else:
                     #         barplot_annotate_brackets(0, 1, pv1.pvalue , np.arange(1,3) , heights,ax,col="blue")
                     # else:
                     #     if pv2.pvalue < 0.05:
                     #         barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax,col="red")
                     #     else:
                     #         pass
                     #         barplot_annotate_brackets(0, 1, r"\noindent HS=LS?: {:.2f}  \\ \\  HS$>$LS?: {:.2f}".format(pv2.pvalue,pv1.pvalue), np.arange(1,4) , heights,ax,barh=0.12)
                    if save:
                         fig.tight_layout() 
                         plt.savefig(r"./Paper_plots/EMG_{}ph_for_{}.pdf".format(phase,mt_name))
                         # barplot_annotate_brackets(0, 1, f"{pv2.pvalue}" , np.arange(1,3) , heights,ax)
                         # if pv2.pvalue < 0.05:
                         #     barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax)
        elif style=="box_cont_ipsi":
                #Combine this with the minimal configuration
                #angs=ANG_plotmultip(parts,ref="head_rel",style="table")
                cond_sset=[0,1,2,3] #excluding MS    The order affect the statistical result
                col=["white","lightgray","darkgray","dimgray"]
                patches= [Patch(facecolor=cl , label=cnd ,edgecolor="black") for cl , cnd in zip(col,["Baseline","Low Stiffness","Middle Stiffness","High Stiffness"])]
                
                cond_sset=[0,1,3] #excluding MS    The order affect the statistical result

                
                side=muscles
                #[f"Right Rot\n(15 deg)" , f"Left Rot\n(15 deg)"]
                
                fig=plt.figure(f"EMG_{met}_{style} group: {phase} phase {Norm} normalization grouping {mGroup}("+",".join(exc_part)+" are excluded)")
                
                if muscN == 0:
                    fig.suptitle(f"EMG_{met}_{style} group: {phase} phase("+",".join(exc_part)+" are excluded)",fontsize = "xx-large")
                    #fig.tight_layout(pad=3, w_pad=0.5, h_pad=0.50) 
                    figs.append(fig)
                    ax = fig.add_subplot()
                   
                
                
                #ax = fig.add_subplot(len(muscles),len(mots),(muscN*len(mots)+mt_ind+1))
                
                #ax.set_title(muscName + " " + mt,fontsize = "xx-large")
                                
                #Formating 
                # ticklabs= [jcond_lb[ind]+"\n"+r" ({}".format(angs[mt_key][mt_ind][ind][0])+"$^{\circ}\pm$"+"{}".format(angs[mt_key][mt_ind][ind][1])+"$^{\circ}$)"for ind in [0,-1]]
                ticklabs= [jcond_lb[i] for i in cond_sset]
                fsx=30
                is_top=False
                is_bottom=not is_top
                
                                     
                
                lableft=True
                if muscN<int(len(muscles)/2):
                    pass
                    #plt.ylim(0,0.6)
                    #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.15))
                else:
                    pass
                    #plt.ylim(0,0.6)
                    #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.15))
                
              
                    
                #ax.tick_params(labeltop=is_top,labelleft=lableft,top=is_top,bottom=is_bottom,labelbottom=is_bottom,right=True)
                plt.yticks(fontsize=30)
                plt.xticks(fontsize=fsx)
                plt.xlim(2.2,14.2)
                
                
                offset=0
                box_pos=np.array([[3*i ,3*i+0.5,3*i+1,3*i+1.5] for i in range(1,1+len(muscles))])+offset
                box_pos=np.array([[3*i ,3*i+0.5,3*i+1] for i in range(1,1+len(muscles))])+offset
                print(box_pos.shape)
                #plt.boxplot(pGs_HS_LS.T , positions=box_pos , labels=ticklabs*len(nmt))
                
                
                #print(pGroups[cond_sset,nmt,:].mean(axis=-1))
                #print(np.percentile(pGroups[cond_sset,nmt,:], 50, axis=-1))
                
                #heights=means[:,nmt,:].max(axis=-1)
                pG_mn=pGroups.mean(axis=-1)
                pG_std=pGroups.std(axis=-1)
                
 
                nmot_in=0
                muscN
                nmot=0
                
                ax.bar(box_pos[muscN], pG_mn[cond_sset,nmot] , yerr=pG_std[cond_sset,nmot] , align='edge', alpha=1, width=-0.5, color=[col[i] for i in cond_sset],edgecolor="black")
               
                
                pairs = [[0,1],[1,2],[2,3]]
                pairs = [[0,1],[1,3]]
                for npair,pair in enumerate(pairs):
                        p1=pair[0]
                        p2=pair[1]
                        p1_in=cond_sset.index(p1)
                        p2_in=cond_sset.index(p2)
                        print(pair)
                      
                        # Significant diferences
                        hyp=0
                        pv=paired_test(pGroups[p1,nmot], pGroups[p2,nmot], hyp)
                        #heights=pGroups[pGind,nmot,:].max(axis=-1)
                        heights=pG_mn+pG_std
                        print(heights.shape)
                        print(pv.pvalue)
                        if pv.pvalue < 0.05:                     
                            # Testing for significant increments
                            hyp=1
                            bar_col="black" 
                            pv=paired_test(pGroups[p1,nmot], pGroups[p2,nmot], hyp)
                            #pv2=paired_test(pGroups_clean[1], pGroups_clean[-1], 2)
                            print()
                            if pv.pvalue < 0.05:
                                    barplot_annotate_brackets(p1_in,p2_in, pv.pvalue , box_pos[muscN]-0.25 , heights[cond_sset,nmot],ax,dh=.07*npair+0.05,col=bar_col)
                            else:
                                    # Testing for significant decrements
                                    hyp=2
                                    bar_col="green"
                                    pv=paired_test(pGroups[p1,nmot], pGroups[p2,nmot], hyp)
                                    if pv.pvalue < 0.05:
                                            barplot_annotate_brackets(p1_in,p2_in , pv.pvalue , box_pos[muscN]-0.25 , heights[cond_sset,nmot],ax,dh=.07*npair+0.05,col=bar_col)           
                if muscN==0:
                    ax.set_xticks(np.arange(1,nMusc+1)*3+0.5,muscles)
                    ax.set_title("Axial Rotation (15 deg)",fontsize = 44)
                    ax.legend(handles=[patches[i] for i in cond_sset],loc="upper center",fontsize=30)
                    ax.set_ylabel("normalized EMG",fontsize=30)
                #barplot_annotate_brackets(0, 1, f"p =st: {pv1.pvalue:.2f}, dt: {pv2.pvalue:.2f}"  , np.arange(1,3) , heights,ax)
                
                 
                 #Stat value to discern betweeen significant increment, decrement or non-significant diference
                 # if pv1.pvalue < 0.05:
                 #     if pv2.pvalue < 0.05:
                 #         barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax,col="green")
                 #     else:
                 #         barplot_annotate_brackets(0, 1, pv1.pvalue , np.arange(1,3) , heights,ax,col="blue")
                 # else:
                 #     if pv2.pvalue < 0.05:
                 #         barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax,col="red")
                 #     else:
                 #         pass
                 #         barplot_annotate_brackets(0, 1, r"\noindent HS=LS?: {:.2f}  \\ \\  HS$>$LS?: {:.2f}".format(pv2.pvalue,pv1.pvalue), np.arange(1,4) , heights,ax,barh=0.12)
                if save:
                     fig.tight_layout() 
                     plt.savefig(r"./Paper_plots/EMG_{}ph_for_{}.pdf".format(phase,mt_name))
                     # barplot_annotate_brackets(0, 1, f"{pv2.pvalue}" , np.arange(1,3) , heights,ax)
                     # if pv2.pvalue < 0.05:
                     #     barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax)
        elif style=="box_minimal_3phases":
            phs=["aproaching","holding","recovery"]
            for mt_name , mt_key in mot_types.items():
                mots=[m for m in motions if m[0] == mt_key]
                print(mots)
                if mt_key == "s":
                    side= ["back" , "front"]
                else:
                    side=["right" , "left"]
                    
                
                aux= [mots[0],mots[-1]]
                mots=aux
                print(mots)
                for mt_ind , mt in enumerate(mots):
                    fig=plt.figure(f"EMG_{met}_{style} group for {mt}: 3 phases")
                    abs_ang= str(np.abs(int(mt[1:])))
                    if muscN == 0:
                        fig.suptitle(f"EMG_{met}_{style} group for {mt_name},{side[mt_ind]} {abs_ang} : 3 phases("+",".join(exc_part)+" are excluded)",fontsize = "xx-large")
                        fig.tight_layout(pad=3, w_pad=0.5, h_pad=0.50) 
                        figs.append(fig)
                    
                    nmt=motions.index(mt)
                    print(phs)
                    ax = fig.add_subplot(len(muscles),3,(muscN*3+phs.index(phase)+1))
                    
                    
                    
                    ax.set_title(muscle + " " + phase ,fontsize = 22)
                    
                    plt.boxplot(means[[0,-1],nmt,:,muscN].T,labels=[jcond_lb[0] , jcond_lb[-1]])
                    plt.yticks(fontsize=20)
                    plt.xticks(fontsize=20)
                    heights=means[[0,-1],nmt,:,muscN].max(axis=-1)
                    print(mt)
                    pGroups_clean=remove_outliers(pGroups[:,nmt,:])
                    pv1=paired_test(pGroups_clean[1], pGroups_clean[-1], 1)
                    pv2=paired_test(pGroups_clean[1], pGroups_clean[-1], 1)
                    
                    if pv1.pvalue < 0.05:
                        if pv2.pvalue < 0.05:
                            barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax,col="green")
                        else:
                            barplot_annotate_brackets(0, 1, pv1.pvalue , np.arange(1,3) , heights,ax,col="blue")
                    else:
                        if pv2.pvalue < 0.05:
                            barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax,col="red")
                        else:
                            barplot_annotate_brackets(0, 1, r"\noindent HS=LS?: {:.2f}  \\ \\  HS$>$LS?: {:.2f}".format(pv2.pvalue,pv1.pvalue), np.arange(1,4) , heights,ax,barh=0.12)
                    if save:
                        fig.tight_layout() 
                        plt.savefig(r"./Paper_plots/EMG_3phs_for_{}.pdf".format(mt))
                        
                    # barplot_annotate_brackets(0, 1, f"{pv2.pvalue}" , np.arange(1,3) , heights,ax)
                    # if pv2.pvalue < 0.05:
                    #     barplot_annotate_brackets(0, 1, pv2.pvalue , np.arange(1,3) , heights,ax)
        elif style=="table":
            stds=stds.mean[:,:,:,muscN](axis = -1)
            means=means.mean[:,:,:,muscN](axis = -1)
            
            heads=[muscle]
            
            for mot in ["Fflex/Rbend" , "Bflex/LBend"]:
                heads+=[f"{mot}_LS"]+ [f"{mot}_HS"] + ["HS != LS?"] + ["HS > LS?"]
            
            EMG_table.append(heads)
            for mt_name , mt_key in mot_types.items():
                EMG_table.append([mt_name])
                
                mots=[m for m in motions if m[0] == mt_key]
                if mt_key == "s":
                    side= ["back" , "front"]
                else:
                    side=["right" , "left"]
                    
                aux= [mots[0],mots[-1]]
                mots=aux
                
                for mt_ind , mt in enumerate(mots):
    
                    nmt=motions.index(mt)
                    print(mt)
    
                    abs_ang= [side[mt_ind]] + [str(np.abs(int(mt[1:])))]
                    
                    
                    pGroups_clean=remove_outliers(pGroups[:,nmt,:])
                    pv1=paired_test(pGroups_clean[1], pGroups_clean[-1], 1)
                    pv2=paired_test(pGroups_clean[1], pGroups_clean[-1], 1)
                    
                    means[0,nmt]
                    [EMG_table[-1].append("{:.2f} +-{:.2f}".format(means[cnd,nmt],stds[cnd,nmt]) ) for cnd in [0,2]]
                    EMG_table[-1].append(pv2.pvalue)
                    EMG_table[-1].append(pv1.pvalue)
            if muscN==3:
                print(tabulate.tabulate(EMG_table, headers = heads, tablefmt="fancy_grid"))  
                with open(r"G:/My Drive/PhD/My research/Experiments/EMG and MOCAP measurements/5- 6 bar rigid blockage, 6 users/Plots/table2.csv","w") as f:
                    writee=csv.writer(f)
                    writee.writerows(EMG_table)
                            
                                
                        
                    
                            
                            
                
        elif style=="lines":
                    for mt_name , mt_key in mot_types.items():
                        mots=[m for m in motions if m[0] == mt_key]
                        
                        fig=plt.figure(f"EMG_{met}_{style} group for {mt_name}: {phase} phase")
                        if muscN == 0:
                            fig.suptitle(f"EMG_{met}_{style} group for {mt_name}: {phase} phase {Norm} normalization ("+",".join(exc_part)+" are excluded)",fontsize = "xx-large")
                            fig.tight_layout(pad=3, w_pad=0.5, h_pad=0.50) 
                            figs.append(fig)
                            
                            
                        for mt_ind , mt in enumerate(mots):
                            nmt=motions.index(mt)
                            ax = fig.add_subplot(len(muscles),len(mots),(muscN*len(mots)+mt_ind+1))
                            ax.set_title(muscName + " " + mt,fontsize = "xx-large")
                            ax.set_xticks(np.arange(len(jcond_lb)),jcond_lb)
                            
                            legs=[f"P{ap+1}" for ap in act_part]
                            ax.plot(means[:,nmt,:,muscN],label=legs)
                            if muscN == 0 and nmt==0:
                                ax.legend()
                                
        elif style=="lines_minimal":
                    for mt_name , mt_key in mot_types.items():
                        mots=[m for m in motions if m[0] == mt_key]
                        
                        #fig=plt.figure(f"EMG_{met}_{style} group for {mt_name}: {phase} phase {Norm} normalization")
                        fig=plt.figure(f"EMG_{met}_{style} group for {mt_name}2: {phase} phase {Norm} normalization grouping {mGroup}("+",".join(exc_part)+" are excluded)")
                        
                        if muscN == 0:
                            fig.suptitle(f"EMG_{met}_{style} group for {mt_name}: {phase} phase ("+",".join(exc_part)+" are excluded)",fontsize = "xx-large")
                            fig.tight_layout(pad=3, w_pad=0.5, h_pad=0.50) 
                            figs.append(fig)
                            
                        aux= [mots[0],mots[-1]]
                        mots=aux    
                        rng=np.arange(3)!=1
                        for mt_ind , mt in enumerate(mots):
                            nmt=motions.index(mt)
                            ax = fig.add_subplot(len(muscles),len(mots),(muscN*len(mots)+mt_ind+1))
                            ax.set_title(muscle + " " + mt,fontsize = "xx-large")
                            ax.set_xticks(np.arange(3)[rng],[jcond_lb[0],jcond_lb[2]])
                            
                            legs=[f"P{ap+1}" for ap in act_part]
                            
                            ax.plot(means[rng,nmt,:,muscN],label=legs)
                            if muscN == 0 and mt_ind==0:
                                ax.legend(loc="upper right")
   
            
            
            
           # ax.update(hspace=0.5)
    
    #mng = plt.get_current_fig_manager()
    #mng.full_screen_toggle()  
         
    # print("Means matrix")
    # print(means_t[:,:,act_part])
    if save:
                plt.savefig(r"./Plots/9302023/EMG_{}_{} group {} phase.pdf".format(met,style,phase))


    
def barplot_annotate_brackets(num1, num2, data, center, height, ax, yerr=None, dh=.05, barh=.05, fs=30, maxasterix=None , col="black"):
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
    

    ry = height[num2]
    rx = center[num2]
    
    

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh
    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y-barh)
    mid = ((lx+rx)/2, y+barh)

    ax.plot(barx, bary, c=col)

    kwargs = dict(ha='center', va='bottom' ,color=col)
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


