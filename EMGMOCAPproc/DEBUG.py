# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:49:33 2023

@author: UTAIL
"""
import sys

EMGMOCAPpt =r"C:\Users\UTAIL\OneDrive\Documents\GitHub\MOCAP-EMG-proc"
if not sys.path.count(EMGMOCAPpt):
    sys.path.append(EMGMOCAPpt)
    
import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

from EMGMOCAPproc import EMG

from scipy.signal import hilbert

import pandas

import hampel

def RPYvsTime(trial,shaded=False,cond="head_abs"):
    RPY = trial.RPY[cond]
    time= np.arange(RPY.shape[1])/trial.sfmc
           
    plt.figure(trial.label + cond)  
    plt.plot(time,RPY.T)
    plt.title(r"{:s}: ".format(trial.label)) 
    
    Xl = r'time' 
    Yl= [r"$\alpha_{sag}$" , r"$\beta_{cor}$" , r"$\gamma_{ax}$"]
    #leg=[r"h_{rel}","h_{abs}","b_{abs}"]  
    plt.xlabel(Xl)
    plt.legend(Yl)
    
    # Adding shaded areas between x-limits
    if hasattr(trial, 'Tbound') and isinstance(trial.Tbound, dict) and shaded:
        color_wheel = mcolors.get_named_colors_mapping()
        key_colors = list(color_wheel.keys())
        
        i_mot=0
        for key, bounds in trial.Tbound.items():
            colors = key_colors[i_mot:i_mot+2]
            i_mot+=1
            flexlims= bounds[0].reshape((-1,2))[:,0]
            extlims = bounds[1].reshape((-1,2))[:,1]
            
            for flim, elim in zip(flexlims,extlims):
                lab_xpos = (flim+elim)/2
                print(lab_xpos)
                lab_ypos = 40
                plt.text(lab_xpos, lab_ypos, key,fontsize= 5)
            for i in range(2):
                color = colors[i % len(colors)]
                limits= bounds[i,:]
                limits= limits.reshape(-1,2)
                for n_lim in range(limits.shape[0]):
                    plt.axvspan(limits[n_lim, 0], limits[n_lim,1], facecolor=color, alpha=0.3)


def EMGvsTime(trial,channels=[0,8,9,10],filt=True,shaded=False,RPY_add=False,Filt_type=1):
    muscles = ['(R) Sternocleidomastoid', '(L) Sternocleidomastoid','(R) Para spinal','(L) Para spinal']
    
    tags= [r"EMG" , r"$\alpha_{sag}$" , r"$\beta_{cor}$" , r"$\gamma_{ax}$"]
    #leg=[r"h_{rel}","h_{abs}","b_{abs}"]  

    
    part_N=trial.label[:-4]
    chan = np.array(channels, dtype=np.intp)
    EMGs = trial.EMG[chan] 
    if filt:
        print("The signal was filtered")
        if Filt_type ==1:
            EMGs=EMG.EMG_filt(EMGs,trial.sfemg)
            fig=plt.figure("EMG vs time "+trial.label+"filt avg")
        else:
            EMGs=EMG.EMG_filt2(EMGs,trial.sfemg)
            fig=plt.figure("EMG vs time "+trial.label+"filt lp")
        #EMGs=EMG.moving_average(np.abs(EMGs),300)
        
    else:
        #for emg in EMGs:
            #print(emg)
            #emg=hampel.hampel(pandas.Series(emg), window_size=301, n=3)
        #EMGs=EMG.moving_average(np.abs(EMGs),300)
        fig=plt.figure("EMG vs time "+trial.label)
      
    
    
    col_idx = np.linspace(0,255,len(trial.mot)).astype(int)
    col = list(mpl.cm.gist_rainbow(col_idx))
    col_dic=dict()
    for c,m in zip(col,trial.mot):
        col_dic.update({m:c})
        
    for m_n , muscle in enumerate(muscles):
        time= np.arange(EMGs.shape[1])/trial.sfemg 
        ax = fig.add_subplot(len(muscles),1,m_n+1)
        ax.set_title(part_N+ ": " + muscle)
        
        ax.plot(time,EMGs[m_n],color="b")
        if RPY_add:
            cond="head_abs"
            RPY = trial.RPY[cond]
            RPY/= RPY.max(axis=1).reshape(-1,1)
            RPY*=max(EMGs[m_n])
            timeRPY= np.arange(RPY.shape[1])/trial.sfmc
            ax.plot(timeRPY,RPY.T)
            if m_n==2:
                ax.legend(tags)
        
        ax.axhline(y=0, color="r", linestyle='--')
        
        
        #ax.plot(time,np.abs(hilbert(EMGs[m_n])),color="r")
        # Adding dotted vertical axis for the limits
        if hasattr(trial, 'Tbound') and isinstance(trial.Tbound, dict) and shaded:
            Tmx= trial.Tbound[trial.seq_from_IMU()[-1]].max()+1
            ax.set_xlim([0,Tmx])
            for key, bounds in trial.Tbound.items():
                flexlims= bounds[0].reshape((-1,2))[:,0]
                extlims = bounds[1].reshape((-1,2))[:,1]
                
                for flim, elim in zip(flexlims,extlims):
                    plt.axvspan(flim,elim, facecolor=col_dic[key], alpha=0.3)
                    if m_n==1:
                        if key=="r29":
                            print()
                        
                        pass
                        #print(f"Mean activity of {muscle} for {key}")
                        #print(EMGs[m_n,int(flim*2000):int(elim*2000)].mean())
                    lab_xpos = (flim+elim)/2
                    lab_ypos = 0.7*max(EMGs[m_n])
                    ax.text(lab_xpos, lab_ypos, key ,fontsize= 10)
        
    print(f" Maximum time: {max(time)}")
    return EMGs.max(axis=1)



def check_bound(parts): 
    for nnp,p in parts.items():
        print(nnp)
        for nc,c in enumerate(p):
             if c.Tbound==None:
                 cond = "empty"
             else:
                    cond = "Full"
             print(f"Tbound in Trial [nc] is {cond}")

def check_IMUd(parts): 
    for nnp,p in parts.items():
        print(nnp)
        for nc,c in enumerate(p):
             if c.IMU==None:
                 cond = "empty"
             else:
                    cond = "Full"
             print(f"IMU in Trial [nc] is {cond}")
