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
import scipy
import tabulate

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

def EMG_plot1p(part_conds,stage="mean_full",save=False,debug=False, channels=[0,8,9,10]):
    muscles = ['(R) Sternocleidomastoid', '(L) Sternocleidomastoid','(R) Para spinal','(L) Para spinal']
    #print(part_conds[0])
    motions = part_conds[0].mot
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
    fig = plt.figure(f"EMG_stat_{part_N} all trials {stage}")
    print(f"Muscle activity of the participant {int(part_N[1:4])}")
    for m_n , muscle in enumerate(muscles):
            #print(f"Processsing {muscle}")
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
            means=np.empty((0,len(motions)))
            stds=np.empty((0,len(motions)))
            
            
            max_EMG_cnd=np.array([EMG.max_EMG(part_conds[i],channels=channels) for i in range(len(part_conds))])
            max_EMG_abs=max_EMG_cnd.max(axis=0)
           
            for n_cond , cond_dat in enumerate(part_conds):
                
                means_stds=EMG.EMG_means(cond_dat,stage,Norm=max_EMG_abs,channels=channels)[:,m_n,:]

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
            for n_cond , cond_lab in enumerate(jcond_lb):    
                EMG_table.append([cond_lab]+["{:.4f} +-{:.4f}".format(m,s) for m , s in zip(means[n_cond],stds[n_cond])])
            print(muscle)
            print(tabulate.tabulate(EMG_table, headers = motions, tablefmt="fancy_grid"))
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
                ax.legend(handles=patches,bbox_to_anchor=(1.2, 0.))
            plt.grid("on")
            ax.set_ylim([0, max(abs(means))*1.1])
           # ax.update(hspace=0.5)
    fig.tight_layout()        
    if save:
                plt.savefig(r"Plots/{} EMG profile {}.pdf".format(part_N,stage))


def EMG_plotmultip(parts,stage="mean_full",save=False,debug=False, Norm=None, channels=[0,8,9,10]):
    muscles = ['(R) Sternocleidomastoid', '(L) Sternocleidomastoid','(R) Para spinal','(L) Para spinal']
    Nparts= len(parts)
    
    motions = parts["P 3"][0].mot
    Nmot= len(motions)
    cond_lb = ['LS1', 'LS2', 'MS1', 'MS2', 'HS1', 'HS2']
    jcond_lb= ['LS', 'MS', 'HS']
    
    fig = plt.figure(f"EMG_stat_ all parts: {stage}")
    for m_n , muscle in enumerate(muscles):
            #print(f"Processsing {muscle}")
            ax = fig.add_subplot(len(muscles),1,m_n+1)
            ax.set_title("All parts:"  + muscle + " " + stage)
            ax.set_xticks(np.arange(len(motions)),motions)
            col_idx = np.linspace(0,255,len(jcond_lb)).astype(int)
            col = list(mpl.cm.gist_rainbow(col_idx))
            patches= [Patch(color=cl , label=cnd) for cl , cnd in zip(col,jcond_lb)]
            
            # Pick the statistics values for each condition
            Ncond=len(jcond_lb)    
            
            means_t=np.zeros((Ncond*2,Nmot,Nparts))
            stds_t=np.zeros((Ncond*2,Nmot,Nparts))
            means=np.empty((0,len(motions)))
            stds=np.empty((0,len(motions)))
            for  part_lab, part_conds in parts.items():
                max_EMG=EMG.max_EMG(part_conds[-1])
                npa=int(part_lab[-1])-2
                for n_cond , cond_dat in enumerate(part_conds):
                    if part_lab == "P 2":
                        pass
                       #print(part_lab + cond_lb[n_cond])
                       #print(n_cond)
                       #print(Norm[npa])
                    means_stds=EMG.EMG_means(cond_dat,stage,Norm=max_EMG, channels=channels)[:,m_n,:]
                  
                    
                    for nm_loc, m in enumerate(cond_dat.mot):
                        nm_glb=motions.index(m)
                        means_t[n_cond,nm_glb,npa] = means_stds[nm_loc,0]
                    
                        
                        stds_t[n_cond,nm_glb,npa] = means_stds[nm_loc,1]
                        
                        if debug:
                            if m_n ==1 and n_cond==0:
                                    pass
                                    #print(f"Mean activity of {muscle} for {m}")
                                    #print(means_stds[nm_loc,0])
                for condf in jcond_lb:
                    trial1=cond_lb.index(f"{condf}1")
                    trial2=cond_lb.index(f"{condf}2")
                    means_t[trial2,means_t[trial2,:,npa]==0,npa] = means_t[trial1,means_t[trial2,:,npa]==0,npa]
                    means_t[trial1,means_t[trial1,:,npa]==0,npa] = means_t[trial2,means_t[trial1,:,npa]==0,npa]
                
                    
                #means=np.vstack((means,means_t[trial2,:]))
                #stds=np.vstack((stds,stds_t[trial2,:]))
            pVals=means_t.reshape(2,Ncond,Nmot,Nparts).mean(axis = 0)[np.arange(3)!=1]
            pTest=scipy.stats.ttest_ind(pVals[0],pVals[1],axis=-1 , alternative="two-sided")
            mr29=means_t.reshape(2,Ncond,Nmot,Nparts).mean(axis = 0)
            iii=motions.index("r-29")
            #print(muscles[m_n])
            #print(mr29[:,iii,:])
            means=means_t.reshape(2,Ncond,Nmot,Nparts).mean(axis = 0).mean(axis = -1)
            stds=stds_t.reshape(2,Ncond,Nmot,Nparts).mean(axis = 0).mean(axis = -1)
            #print(condf)
            #print(means_t[trial1,:])
            #print(means_t[trial2,:])
            #print(means[-1,:])
           
            means=means.T.reshape(-1)
            stds=stds.T.reshape(-1)
            ##print(f"means: {means}")
            # Calculate the x positions for the bars
            x = np.array([np.linspace(0,0.5,Ncond) + i for i in range(len(motions))]).reshape(-1)
            
            for npv , pv in enumerate(pTest.pvalue):
                barplot_annotate_brackets(npv*3 , npv*3+2, f"p = {pv:.2f}" , x , abs(means),ax)
            
            # Plot the means as bars
            ##print(abs(means))
            ax.bar(x, abs(means), yerr=stds, align='center', alpha=0.7, width=0.5/Ncond, color=col*Ncond)
            if m_n==2:
                ax.legend(handles=patches,bbox_to_anchor=(1.2, 0.))
            plt.grid("on")
            ax.set_ylim([0, max(abs(means))*1.1])
           # ax.update(hspace=0.5)
    fig.tight_layout()        
    if save:
                plt.savefig("Plots/All part EMG profile {}.pdf".format(stage))

    
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

def filter_data(MC_EMG):
    
    for trial in part_MC_EMG:
        plt.figure()
        RPY = trial.RPY["body_abs"]
        # Select the desired array to filter and plot (e.g., index 0 for the first array)
        # Define the filter parameters
        order = 4  # Filter order
        cutoff_freqs = np.linspace(0.001, 0.01, num=10)  # Cutoff frequencies for filtering
        # Define the vertical shift amount for visualization
        shift_amount = 5.0
        # Apply low-pass filter to the curve with increasing cutoff frequencies
        for i, cutoff_freq in enumerate(cutoff_freqs):
            # Design the Butterworth low-pass filter
            b, a = butter(order, cutoff_freq, btype='low', analog=False, output='ba')
            # Apply the filter to the curve
            filtered_curve = filtfilt(b, a, RPY[2, :])
            # Calculate the vertical shift value
            shift_value = i * shift_amount
            # Plot the filtered curve with vertical shift
            plt.plot(filtered_curve + shift_value)
            
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()  
        plt.xlabel('Column')
        plt.ylabel('Value')
        plt.title('Low-Band Filtered Curves with Vertical Shift: {}'.format(trial.label))
        plt.show()
