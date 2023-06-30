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

import matplotlib.pyplot as plt

import matplotlib.colors as mcolors



def RPYvsTime(trial,shaded=False):
    RPY = trial.RPY["head_abs"]
    time= np.arange(RPY.shape[1])/trial.sfmc
           
    plt.figure(trial.label)  
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
            for i in range(2):
                color = colors[i % len(colors)]
                limits= bounds[i,:]
                limits= limits.reshape(-1,2)
                for n_lim in range(limits.shape[0]):
                    plt.axvspan(limits[n_lim, 0], limits[n_lim,1], facecolor=color, alpha=0.3)
                    


