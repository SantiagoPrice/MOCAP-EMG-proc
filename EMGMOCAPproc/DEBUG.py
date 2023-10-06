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

import scipy

def RPYvsTime(trial,shaded=False,cond="head_abs"):
    RPY = trial.RPY[cond]
    time= np.arange(RPY.shape[1])/trial.sfmc
           
    plt.figure(trial.label + cond)  
    plt.plot(time,RPY.T)
    plt.title(r"Rotation assesment for{:s} cond {} ".format(trial.label,cond)) 
    
    Xl = r'time' 
    Yl= [r"$\alpha_{sag}$" , r"$\beta_{cor}$" , r"$\gamma_{ax}$"]
    #leg=[r"h_{rel}","h_{abs}","b_{abs}"]  
    plt.xlabel(Xl)
    plt.legend(Yl)
    plt.axhline(y=0,linestyle="--",color="red")
    
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


def EMGvsTime_alltrials(part,chans=[0,8,9,10],f=True,s=False,rpy=False,filt=1):
    for trial in part:
        if trial.label[-2] !="F": 
            print(f"Adding {trial.label}")
            EMGvsTime(trial,channels=chans,filt=f,shaded=s,RPY_add=rpy,Filt_type=filt,superp=True,spec=False)

def EMGvsTime(trial,channels=[0,8,9,10],filt=True,shaded=False,RPY_add=False,Filt_type=1,superp=False,spec=False):
    muscles = ['(R) Sternocleidomastoid', '(L) Sternocleidomastoid','(R) Para spinal','(L) Para spinal']
    
    tags= [r"EMG" , r"$\alpha_{sag}$" , r"$\beta_{cor}$" , r"$\gamma_{ax}$"]
    #leg=[r"h_{rel}","h_{abs}","b_{abs}"]  

    
    
    part_N=trial.label[:-4]
    chan = np.array(channels, dtype=np.intp)
    EMGs = trial.EMG[chan]
    
    
    if superp:
        Fig_name_head="EMG vs time "+" ".join(trial.label.split(" ")[:2])
    else:
        Fig_name_head="EMG vs time "+trial.label
    
    if filt:
        print("The signal was filtered")
        
        if Filt_type ==1:
            EMGs=EMG.EMG_filt(EMGs,trial.sfemg)
            fig=plt.figure(Fig_name_head+"filt avg")
        elif Filt_type ==2:
            EMGs=EMG.EMG_filt2(EMGs,trial.sfemg)
            fig=plt.figure(Fig_name_head+"filt lp")
        elif Filt_type ==3:
            EMGs=EMG.EMG_filt3(EMGs,trial.sfemg)
            fig=plt.figure(Fig_name_head+"filt bp + mavg")
            
        #EMGs=EMG.moving_average(np.abs(EMGs),300)
        
    else:
        #for emg in EMGs:
            #print(emg)
            #emg=hampel.hampel(pandas.Series(emg), window_size=301, n=3)
        #EMGs=EMG.moving_average(np.abs(EMGs),300)
        fig=plt.figure("EMG vs time "+trial.label)
      
    ax_list=[ax.get_label()  for ax in fig.axes]
    
    
    col_idx = np.linspace(0,255,8).astype(int)
    col_line = list(mpl.cm.nipy_spectral(col_idx))
    
    for m_n , muscle in enumerate(muscles):
        time= np.arange(EMGs.shape[1])/trial.sfemg 
        if muscle in ax_list:
            ax=fig.axes[ax_list.index(muscle)]
            
        else:
            ax = fig.add_subplot(len(muscles),1,m_n+1,label=muscle)
            ax.set_title(part_N+ ": " + muscle)
            ax.axhline(y=0, color="r", linestyle='--',label="haxis")

        ax.plot(time,EMGs[m_n],color=col_line[len(ax.get_lines())] , label=trial.label[4:])
        
        if spec:
            fig_spec=plt.figure(fig.get_label()+muscle+"spectogram")
            f, t, Sxx =scipy.signal.spectrogram(EMGs[m_n]/EMGs[m_n].min(axis=0),fs=trial.sfemg, window=('tukey', 1))
            
            print(muscle)
            print((EMGs[m_n]/EMGs[m_n].min(axis=0)).min())
            print(Sxx.shape)
            print(f"max comp {Sxx.argmax(axis=0)}")
            print(f"min comp{Sxx.argmin(axis=0)}")
            axi = fig_spec.add_subplot(111)
            axi.set_title(part_N+ ": " + muscle)
            

            #axi.plot(f[70:],Sxx[70:,int(150*trial.sfemg/2000)])
            plt.pcolormesh(t, f[70:], Sxx[70:], shading='gouraud',label="spectogram",figure=fig_spec)
            
        if muscle in ax_list and m_n==1:
                lines= [ln for ln in ax.get_lines() if ln.get_label() != "haxis" ]
                ax.legend(handles=lines)
        
        
        if RPY_add:
            cond="head_abs"
            RPY = trial.RPY[cond]
            RPY/= RPY.max(axis=1).reshape(-1,1)
            RPY*=max(EMGs[m_n])
            timeRPY= np.arange(RPY.shape[1])/trial.sfmc
            ax.plot(timeRPY,RPY.T)
            if m_n==2:
                ax.legend(tags)
        
        
        
        col_idx = np.linspace(0,255,len(trial.mot)).astype(int)
        col = list(mpl.cm.gist_rainbow(col_idx))
        col_dic=dict()
        for c,m in zip(col,trial.mot):
            col_dic.update({m:c})
        #ax.plot(time,np.abs(hilbert(EMGs[m_n])),color="r")
        # Adding dotted vertical axis for the limits
        if hasattr(trial, 'Tbound') and isinstance(trial.Tbound, dict) and shaded:
            #Tmx= trial.Tbound[trial.seq_from_IMU()[-1]].max()+1
            Tmx= np.array([times.max(axis=0).max() for times in trial.Tbound.values()]).max()

            ax.set_xlim([0,Tmx])
            for key, bounds in trial.Tbound.items():
                flexlims= bounds[0].reshape((-1,2))
                extlims = bounds[1].reshape((-1,2))
                
                for flim, elim in zip(flexlims,extlims):           
                    plt.axvspan(flim[0],elim[1], facecolor=col_dic[key], alpha=0.3)
                    
                    plt.axvline(flim[1], color="r")
                    plt.axvline(elim[0], color="r")
                    if m_n==1:
                        if key=="r29":
                            print()
                        
                        pass
                        #print(f"Mean activity of {muscle} for {key}")
                        #print(EMGs[m_n,int(flim*2000):int(elim*2000)].mean())
                    lab_xpos = (flim[1]+elim[0])/2
                    lab_ypos = 0.7*max(EMGs[m_n])
                    ax.text(lab_xpos, lab_ypos, key ,fontsize= 10)
        
    #print(f" Maximum time: {max(time)}")
    return fig



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
