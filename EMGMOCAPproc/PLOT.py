# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 18:12:09 2023

@author: UTAIL
"""
import sys
EMGMOCAPpt =r"C:\Users\UTAIL\OneDrive\Documents\GitHub\MOCAP-EMG-proc"
if not sys.path.count(EMGMOCAPpt):
    sys.path.append(EMGMOCAPpt)
from EMGMOCAPproc import EMG   

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib.patches import Patch

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

def ang_stats(part_trials,participant="" , save = False):

    
    motions = part_trials[0].mot
    cond = ["Low Stiff Part1","Low Stiff Part2","Medium Stiff Part1","Medium Stiff Part2","High Stiff Part1","High Stiff Part1"]
    planes = ["s","l","r"]
    
    fig = plt.figure()
    
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Angles for diferent conditions" + participant)
    
    
    ax.set_xticks(np.arange(len(motions)),motions)
    ax.set_yticks([-40,-29,-13,-9,9,13,29,40])
    color_wheel = mcolors.get_named_colors_mapping()
    color_key = list(color_wheel.keys())
    col = [color_key[i*50] for i in range(len(cond))]
    bar_width = 1/(len(part_trials)*2)
              
    for i , mot in enumerate(motions):
        index= planes.index(mot[0])
        means_stds=np.empty((2,0))
        print(mot , index)
        # Pick the statistics values for each condition
        for trial in part_trials:
            
            offset=trial.get_mean_ang()["value"][:,index]
            
            angs=trial.get_mean_ang(neutral=False)
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

def EMG_plot1p(part_conds,stage="mean_full",save=False):
    muscles = ['(R) Sternocleidomastoid', '(L) Sternocleidomastoid','(R) Para spinal','(L) Para spinal']
    print(part_conds[0])
    motions = part_conds[0].mot
    part_N= part_conds[0].label
    cond_lb = ['LS1', 'LS2', 'MS1', 'MS2', 'HS1', 'HS2']
    jcond_lb= ['LS', 'MS', 'HS']
    
    fig = plt.figure()
    for m_n , muscle in enumerate(muscles):
            print(f"Processsing {muscle}")
            ax = fig.add_subplot(len(muscles),1,m_n+1)
            ax.set_title(part_N[:-4]+ ": " + muscle + " " + stage)
            ax.set_xticks(np.arange(len(motions)),motions)
            col_idx = np.linspace(0,255,len(jcond_lb)).astype(int)
            col = list(mpl.cm.gist_rainbow(col_idx))
            patches= [Patch(color=cl , label=cnd) for cl , cnd in zip(col,jcond_lb)]
            
            # Pick the statistics values for each condition
            Ncond=len(jcond_lb)    
            
            means_t=np.zeros((Ncond*2,len(motions)))
            stds_t=np.zeros((Ncond*2,len(motions)))
            means=np.empty((0))
            stds=np.empty((0))
            
            for n_cond,cond_dat in enumerate(part_conds):
                means_stds=EMG.EMG_means(cond_dat,stage)[:,m_n,:]
                for nm_loc, m in enumerate(cond_dat.mot):
                    nm_glb=motions.index(m)
                    means_t[n_cond,nm_glb] = means_stds[nm_loc,0]
                    stds_t[n_cond,nm_glb] = means_stds[nm_loc,1]
            for condf in jcond_lb:
                trial1=cond_lb.index(f"{condf}1")
                trial2=cond_lb.index(f"{condf}2")
                means=np.hstack((means,means_t[trial1,:]))
                stds=np.hstack((stds,stds_t[trial1,:]))
                #means=np.hstack((means,means_t[trial1:trial2,:].mean(axis = 0)))
                #stds=np.hstack((stds,stds_t[trial1:trial2,:].mean(axis = 0)))
                
            means=means.reshape(-1)
            stds=stds.reshape(-1)
            #print(f"means: {means}")
            # Calculate the x positions for the bars
            x = np.array([np.linspace(0,0.5,Ncond) + i for i in range(len(motions))]).reshape(-1)
            
            # Plot the means as bars
            #print(abs(means))
            ax.bar(x, abs(means), yerr=stds, align='center', alpha=0.7, width=0.5/Ncond, color=col*Ncond)
            if m_n==2:
                ax.legend(handles=patches,bbox_to_anchor=(1.2, 0.))
            plt.grid("on")
            ax.set_ylim([0, max(abs(means))*1.1])
           # ax.update(hspace=0.5)
    fig.tight_layout()        
    if save:
                plt.savefig(r"Plots/{} EMG profile {}.pdf".format(part_N,stage))