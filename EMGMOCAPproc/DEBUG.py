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

import matplotlib.ticker as ticker

from EMGMOCAPproc import EMG

from scipy.signal import hilbert

import pandas

import hampel

import scipy


def RPY_and_EMG(trial,channels=[0,8,9,10],shaded=True,RPY_add=False,Filt_type=1,superp=False,spec=False,plt_type="Tser"):
    figEMGRPY=plt.figure("RPY and EMG")
    axEMG=figEMGRPY.add_subplot(211,label="right SPL")
    axRPY=figEMGRPY.add_subplot(212,sharex=axEMG)
    
    RPYvsTime(trial,shaded,cond="head_abs",AX=axRPY,FIG=figEMGRPY)
    EMGvsTime(trial,channels,shaded,RPY_add,Filt_type,superp,spec,plt_type,AX=axEMG,FIG=figEMGRPY)
    plt.yticks(fontsize=30)
    plt.xticks(fontsize=30)
    return figEMGRPY
    
    

def RPYvsTime(trial,shaded=False,cond="head_rel",AX=None,FIG=None):
    RPY = trial.RPY[cond]
    time= np.arange(RPY.shape[1])/trial.sfmc
    if FIG:
        fig=FIG
    else:
        fig=plt.figure(trial.label + cond) 
    if AX:
        ax=AX
        ax.set_title(r" roll,pitch,yaw angles of the head during the trial")
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title(r"Rotation assesment for{:s} cond {} ".format(trial.label,cond))
    ax.plot(time,RPY.T,linewidth=2)
    
    
    Xl = r'time' 
    Yl= [r"$\alpha_{sag}$" , r"$\beta_{cor}$" , r"$\gamma_{ax}$"]
    #leg=[r"h_{rel}","h_{abs}","b_{abs}"]  
    plt.xlabel(Xl)
    plt.legend(Yl,fontsize=30,loc="lower left")
    plt.axhline(y=0,color="black")
    
    

    
    
    if hasattr(trial, 'Tbound') and isinstance(trial.Tbound, dict) and shaded:
        # Adding shaded areas between x-limits
        # Adding shades to diferentiate motions
        col_idx = np.linspace(0,255,len(trial.mot)).astype(int)
        col = list(mpl.cm.gist_rainbow(col_idx))
        col_dic=dict()
        for c,m in zip(col,trial.mot):
            col_dic.update({m:c})
        #ax.plot(time,np.abs(hilbert(EMGs[m_n])),color="r")
        # Adding dotted vertical axis for the limits
        
        #Tmx= trial.Tbound[trial.seq_from_IMU()[-1]].max()+1
        Tmx= np.array([times.max(axis=0).max() for times in trial.Tbound.values()]).max()

        ax.set_xlim([0,Tmx])
        ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        plt.yticks(fontsize=30)
        plt.xticks(fontsize=30)
                                   
        for key, bounds in trial.Tbound.items():
            
            #relabeling
            mtp=key[0]
            ang=int(key[1:])
            ang_rounded=int(np.round(abs(ang)/5)*5)
            tag_head_remap={"s":["BF","FF"],"r":["RR","LR"],"l":["RF","LF"],}
            tag=tag_head_remap[mtp][np.sign(ang)>0]+str(ang_rounded)
            
            flexlims= bounds[0].reshape((-1,2))
            extlims = bounds[1].reshape((-1,2))
            
            for flim, elim in zip(flexlims,extlims):           
                ax.axvspan(flim[0],elim[1], facecolor=col_dic[key], alpha=0.2)
                
                ax.axvline(flim[1], color="r" ,linewidth=0.7,linestyle=(0, (5, 5)))
                ax.axvline(elim[0], color="r" ,linewidth=0.7,linestyle=(0, (5, 5)))
                    #print(f"Mean activity of {muscle} for {key}")
                    #print(EMGs[m_n,int(flim*2000):int(elim*2000)].mean())
                lab_xpos = (flim[1]+elim[0])/2-4
                lab_ypos = 0.7*(RPY.T).max().max()
                ax.text(lab_xpos, lab_ypos, tag ,fontsize= 30)  
    
    # if hasattr(trial, 'Tbound') and isinstance(trial.Tbound, dict) and shaded:
    #     color_wheel = mcolors.get_named_colors_mapping()
    #     key_colors = list(color_wheel.keys())
        
    #     i_mot=0
    #     for key, bounds in trial.Tbound.items():
    #         colors = key_colors[i_mot:i_mot+2]
    #         i_mot+=1
    #         flexlims= bounds[0].reshape((-1,2))[:,0]
    #         extlims = bounds[1].reshape((-1,2))[:,1]
            
    #         for flim, elim in zip(flexlims,extlims):
    #             lab_xpos = (flim+elim)/2
    #             print(lab_xpos)
    #             lab_ypos = 40
    #             plt.text(lab_xpos, lab_ypos, key,fontsize= 5)
    #         for i in range(2):
    #             color = colors[i % len(colors)]
    #             limits= bounds[i,:]
    #             limits= limits.reshape(-1,2)
    #             for n_lim in range(limits.shape[0]):
    #                 plt.axvspan(limits[n_lim, 0], limits[n_lim,1], facecolor=color, alpha=0.3)
    fig.subplots_adjust(left=0.013, right=0.996 , top=0.98 , bottom=0.027, wspace=0.2 , hspace=0.2)
    return fig
def crit(a):
    return ord(a[0])*100+int(a[1:])

def EMGvsTime_alltrials(part,chans=[0,8,9,10],f=True,s=False,rpy=False,filt=1,ts="Tser"):
    #part.reverse()
    part_reversed=part[::-1]
    print("algo")
    for trial in part_reversed:
            if trial.label[-3:-1] =="MS":
                continue
            if trial ==part_reversed[0]:
                ss=True
            else:
                ss=False
            EMGvsTime(trial,channels=chans,shaded=ss,RPY_add=rpy,Filt_type=filt,superp=True,spec=False,plt_type=ts)

def EMGvsTime(trial,channels=[0,8,9,10],shaded=False,RPY_add=False,Filt_type=1,superp=False,spec=False,plt_type="Tser",AX=None,FIG=None):
    muscles = ['(R) Sternocleidomastoid', '(L) Sternocleidomastoid','(R) Para spinal','(L) Para spinal']
    muscles = ['right STR', 'left STR','right SPL','left SPL']
    tags= [r"EMG" , r"$\alpha_{sag}$" , r"$\beta_{cor}$" , r"$\gamma_{ax}$"]
    #leg=[r"h_{rel}","h_{abs}","b_{abs}"]  
    motions = trial.mot
    motions.sort(key=crit)
    print(f"Processing trial {trial.label}")
    
    part_N=trial.label[:-4]
    chan = np.array(channels, dtype=np.intp)
    EMGs = trial.EMG[chan]
    Nmot=len(motions)
    Nmusc=len(muscles)
    muscle_short =  ['R Stern ', 'L Stern ','R ParaSp ','L ParaSp ']
    if superp:
        Fig_name_head="EMG vs time "+" ".join(trial.label.split(" ")[:2])
    else:
        Fig_name_head="2EMG vs time "+trial.label
    
    Fig_name_head+=plt_type
    
        
    if Filt_type == 0:
        #for emg in EMGs:
            #print(emg)
            #emg=hampel.hampel(pandas.Series(emg), window_size=301, n=3)
        #EMGs=EMG.moving_average(np.abs(EMGs),300)
        fig=plt.figure(Fig_name_head)
    elif Filt_type == 1:
        EMGs=EMG.EMG_filt(EMGs,trial.sfemg)
        Fig_name_head+="filt avg"
    elif Filt_type == 2:
        EMGs=EMG.EMG_filt2(EMGs,trial.sfemg)
        Fig_name_head+="filt lp"
    elif Filt_type == 3:
        EMGs=EMG.EMG_filt3(EMGs,trial.sfemg)
        Fig_name_head+="filt bp + mavg"
    elif Filt_type == 4:
        EMGs=EMG.EMG_filt4(EMGs,trial.sfemg)
        Fig_name_head+="filt bp + hamp"
    elif Filt_type == 5:
        EMGs=EMG.EMG_filt5(EMGs,trial.sfemg)
        Fig_name_head+="filt lp +  filt hp +hamp"
    elif Filt_type == 6:
        EMGs=EMG.EMG_filt6(EMGs,trial.sfemg)
        Fig_name_head+="filt bp + rect + mavg (RAL)"
        
    else:
        raise Exception("Wrong filter")
    if FIG:
        fig=FIG
    else:    
        fig=plt.figure(Fig_name_head+"filt bp + mavg")    
    #EMGs=EMG.moving_average(np.abs(EMGs),300)
    
        
      
    ax_list=[ax.get_label()  for ax in fig.axes]
    
    
    col_idx = np.linspace(0,255,30).astype(int)
    col_line = list(mpl.cm.nipy_spectral(col_idx))
    
    for m_n , muscle in enumerate(muscles):
        if AX and m_n !=2:

            continue
        time= np.arange(EMGs.shape[1])/trial.sfemg
        
        #managing figures
        if plt_type =="Tser":
            
            if muscle in ax_list:
                
                ax=fig.axes[ax_list.index(muscle)]
                
            else:
                if AX and m_n ==2:
                    
                   ax=AX
                   axtit="EMG activity of the "+ muscle
                   
                else:
                   ax = fig.add_subplot(Nmusc,1,m_n+1,label=muscle)
                   axtit=part_N+ ": " + muscle
                ax.set_title(axtit)
                ax.axhline(y=0, color="r", linestyle='--',label="haxis")
                ax.axhline(y=0, color="r", linestyle='--',label="haxis")
                axs=[ax]
                
        else:
            if plt_type =="EMGperMot":
                N_rows=Nmusc
                tit_header=muscle_short
            else:
                N_rows=3
                tit_header=["pitch","roll","yaw"]
                if m_n > 2:
                    break
                
                
            axs={}
            if len(ax_list) > 0:
                [axs.update({mt:fig.axes[ax_list.index(muscle+" "+mt)]}) for mt in motions]
            else:    
                [axs.update({mt:fig.add_subplot(N_rows,Nmot,m_n*Nmot+(ind+1),label=muscle+" "+mt,title=tit_header[m_n]+mt)}) for ind ,  mt in enumerate(motions)]
                for graph in axs.values():
                    graph.axhline(y=0, color="r", linestyle='--',label="haxis")
                    graph.axvline(x=3, color="b", linestyle='--',label="vaxis1")
                    graph.axvline(x=9, color="b", linestyle='--',label="vaxis2")
                    graph.set_xticks(np.array([3,9]),["flex2static","static2ext"])
                    


        #uploading figures
        
        if plt_type == "Tser":       
            ax.plot(time,EMGs[m_n],color=col_line[len(ax.get_lines())] , label=trial.label[4:])           
            
            # Adding shades to diferentiate motions
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
                ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
                ax.tick_params(axis='x', labelsize=30)
                ax.tick_params(axis='y', labelsize=30)
                # ax.yticks(fontsize=30)
                # ax.xticks(fontsize=30)                          
                for key, bounds in trial.Tbound.items():
                    
                    #relabeling
                    mtp=key[0]
                    ang=int(key[1:])
                    ang_rounded=int(np.round(abs(ang)/5)*5)
                    tag_head_remap={"s":["BF","FF"],"r":["RR","LR"],"l":["RF","LF"],}
                    tag=tag_head_remap[mtp][np.sign(ang)>0]+str(ang_rounded)
                    
                    flexlims= bounds[0].reshape((-1,2))
                    extlims = bounds[1].reshape((-1,2))
                    
                    for flim, elim in zip(flexlims,extlims):           
                        ax.axvspan(flim[0],elim[1], facecolor=col_dic[key], alpha=0.2)
                        
                        ax.axvline(flim[1], color="r" ,linewidth=0.7,linestyle=(0, (5, 5)))
                        ax.axvline(elim[0], color="r" ,linewidth=0.7,linestyle=(0, (5, 5)))
                        if m_n==1:
                            if key=="r29":
                                print()
                            
                            pass
                            #print(f"Mean activity of {muscle} for {key}")
                            #print(EMGs[m_n,int(flim*2000):int(elim*2000)].mean())
                        lab_xpos = (flim[1]+elim[0])/2-4
                        lab_ypos = 0.7*max(EMGs[m_n])
                        ax.text(lab_xpos, lab_ypos, tag ,fontsize= 30)                      
                        
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
                    
        else: 
           if plt_type=="EMGperMot":
               SF=trial.sfemg
           else:
               SF=trial.sfmc
        
           for key, bounds in trial.Tbound.items():
                   flexlims= (bounds[0].reshape((-1,2))*SF).astype(int)
                   extlims = (bounds[1].reshape((-1,2))*SF).astype(int)
                   
                   for flim, elim in zip(flexlims,extlims):
                       Tremap_flex=np.linspace(0,3,flim[1]-flim[0])
                       Tremap_static=np.linspace(3,9,elim[0]-flim[1])
                       Tremap_ext=np.linspace(9,12,elim[1]-elim[0])
                    
                       time=np.hstack((Tremap_flex,Tremap_static,Tremap_ext))
                       
                       ax=axs[key]
                       cline=col_line[len(ax.get_lines())-3]
                       lb=trial.label[4:]

                       
                       if plt_type=="EMGperMot":
                           EMG_rng=EMGs[m_n,flim[0]:flim[0]+len(time)]
                           ax.plot(time,EMG_rng,color=cline, label=lb)
                       else:
                           cond="head_rel"
                           RPY = trial.RPY[cond]
                           angle=RPY[m_n,flim[0]:flim[0]+len(time)]
                           ax.plot(time,angle,color=cline, label=lb)
           if m_n==0:
                    ax=axs[motions[0]]
                    lines= [ln for ln in axs[motions[0]].get_lines() if not ln.get_label() in ["haxis","vaxis1","vaxis2"]]
                    ax.legend(handles=lines)           
                       
                           #print(f"Mean activity of {muscle} for {key}")
                           #print(EMGs[m_n,int(flim*2000):int(elim*2000)].mean())
               
            
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
