# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 15:35:15 2023

@author: UTAIL
"""
# =============================================================================
# Libraries
# =============================================================================
import sys
import os
EMGMOCAPpt =r"C:\Users\UTAIL\OneDrive\Documents\GitHub\MOCAP-EMG-proc"
if not sys.path.count(EMGMOCAPpt):
    sys.path.append(EMGMOCAPpt)
    
from EMGMOCAPproc import MOCAP ,  DEBUG ,MRKs2REFs ,PLOT ,AUX
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas

# =============================================================================
# Extra parameters
# =============================================================================

plt.rcParams.update({
    "text.usetex": True,
    "font.family":   "Computer Vision"
})
 

# =============================================================================
# Main file
# =============================================================================

# Aux variables
Hmrk = MRKs2REFs.axis_form_square_arrangement_head_EMBC24
Smrk= MRKs2REFs.axis_form_square_arrangement_back_EMBC24

# Control variables
C3dsfolder = r".\c3d files" 
prep_done = True# True if the preprocessing was done
vib_analisis=False # Vibration analisist of one marker on the head and another one on the back
vis_mot = False
EMG_proc= False
EMG_comp= False
set_bound= True
IMU_data = False
ang_comp=False

files = np.arange(7,15)
NPart = 7
C3ds=dict()


# Participants Dataset directory creation
#File name example: 20240129Me1.c3d

partNms = [f"P {i+1}" for i in range(NPart)]
date = ["20240129"]+["20240131"]+["20240127"]+["20240129"]*2+["20240130"]*2
key = ["Me"]*7
ranges=[(83,87)]+[(103,108)]+[(60,64)]+[(65,69)]+[(74,82)]+[(89,93)]+[(95,101)]
exep=[()]+[(105,)]+[()]*2+[(74,76,78,79)]+[()]+[(96,99)]

for i , part in enumerate(partNms):
    if not part=="P 2":
        continue
    
    trials=[]
    for n in range(*ranges[i]):
        if exep[i].count(n):
            continue
        C3d = os.path.join(C3dsfolder , date[i] , key[i]+f'{n:02d}.c3d')
        trials.append([C3d])
    C3ds.update({partNms[i]:trials})
        
    

conds = ["F","L","M","H"]



Modes = ["Free","Low Stiff ","Medium Stiff ","High Stiff"]


 # Uploading participant objects or creating them from scratch (~3min/obj)
if not prep_done:
    #creating object
    parts=dict()
    IMUfold = r"./participant_record" 

    for npart , part_c3d in C3ds.items():
        if  npart== "P 3":
            continue
        print(f"Processing participants data {npart}")


        parts.update({npart:[]})
        
        
        for n , C3d in enumerate(part_c3d):

                             
                     print(f"Part {int(npart[-2:])}_{Modes[n]}.txt")
                     #IMUf= os.path.join(IMUfold,f"Part {int(npart[-2:])}_{M[n]}.txt")
                     args = [C3d, None, npart + " " + conds[n] ,Hmrk,Smrk]                       
                     parts[npart].append(MOCAP.trial_MOCAP_EMG(*args))
             
        with open(f"{npart}_profile.pkl","wb") as f:
                print(f"trials from {npart} were saved")
                pickle.dump(parts[npart],f)
else:
    #uploading object
    try:
        parts #checking if object is already in the console namespace
    except NameError:
        
        parts=dict()
        partNms = [f"P {i}" for i in range(1,NPart+1)] # The forth and third participant are excluded because of the corrupted c3d file
        print(f"Participant list: {partNms}")
        for npart in partNms:
                if npart=="P 4":
                    continue
                with open(f"{npart}_profile.pkl","rb") as f:
                    print(f"loading {npart}")
                    parts.update({npart:pickle.load(f)})
    else:
        "All the participants are loaded"
        pass
                  



        
        


if vis_mot:        
    DEBUG.RPYvsTime(parts["P 1"][0],False)
    DEBUG.RPYvsTime(parts["P 2"][0],False)        
            
         
if ang_comp:
    for npart , trials in parts.items():
            for angle in ["y","r","p"]:
                PLOT.comparingAngle(trials , angle ,"body_abs", participant = npart,save=False)
                PLOT.comparingAngle(trials , angle ,"head_abs",participant = npart,save=False)
                PLOT.comparingAngle(trials , angle ,"head_rel",participant = npart,save=False)

check_ang=False              
if check_ang:
    for npart , trials in parts.items():
            PLOT.ang_stats(trials,participant=npart , save = True)

        
ang_save=False          
if ang_save:
    a=PLOT.savingAngle(parts)
    with open("angles.pkl","wb") as f:
        pickle.dump(a,f)

q_comp=False
if q_comp:
    for frame in ["body","head"]:
        PLOT.comparingQuat(parts,frame)

if set_bound:
        seqsfolder=".\\participant_record"
        final_name = ["F1","L1","M1","H1"]
        
        for npart , trials in parts.items():
            if npart =="P 4":
                continue
            part_name=os.path.join(seqsfolder,npart+"_")
            
            for fn , trial in zip(final_name,trials):
                with open("participant_record/P 1_F1.txt","rb") as f:
                    df=pandas.read_csv(f,skiprows=2,skipfooter=1,header=None)
                    targetsqs=np.array([[elem for elem in row[0].split(" ") if elem != ""] for row in df.iloc],dtype=np.float64)
                    sequ=[]
                    mot_types=["s","r","l"]
                    for q in targetsqs:
                        ang=int(np.arccos(q[0])*2*180/np.pi)
                        axis=np.argmax(np.abs(q[1:]))
                        sign=(-1)**(q[axis+1]<0)
                        sequ.append(f"{mot_types[axis]}{sign*ang}")
                with open("participant_record/P 1_F1.txt","rb") as f:
                    df=pandas.read_csv(f,skiprows=1,nrows=1,header=None)
                    time_phases=np.array([[elem for elem in df.iloc[0]]])
                    time_phases=time_phases[:,[1,2,3,0]]
                print(isinstance(time_phases,np.ndarray))
                print(sequ)
                trial.set_boundaries(sequ,time_phases)
                # with open(f"{npart}_profile.pkl","wb") as f:
                #                     print(f"The sequence for the {fn}-th cond of the {npart} was saved")
                #                     pickle.dump(parts[npart], f)

                # if len(seq) == len(set(seq)):
                #     trial.set_boundaries(seq)
                # else:
                #     print("There are repeating values for {}".format(trial.label))
                    
                # with open(f"{npart}_profile_aux2.pkl", 'wb') as f:
                #             print(f"The sequence for the {n}-th cond of the {npart} was saved")

                #             pickle.dump(parts[npart], f)
        
# # if IMU_data:
# #     IMUfold = r"./participant_record" 
# #     P_list= [f"Part {i+1}" for i in range(NPart)]
# #     Modes = ["Low Stiff Part1","Low Stiff Part2","Medium Stiff Part1","Medium Stiff Part2","High Stiff Part1","High Stiff Part1"]
# #     for n , part in enumerate(P_list):
# #         f"P {n}"
# #         for md in Modes: 
# #             IMUf= sys.path.join(IMUfold,f"{part}{md}")
# #             with open(IMUf,'r') as f:
# #                 data=pandas.read_csv(f)





EMG_comp = False
if EMG_comp:
    for npart , trials in parts.items():
        if npart =="P 1":
            continue
        PLOT.EMG_plot1p_EMBC(trials,chans=[0,1,2,3,4,5],phase="holding")
    
            
                
#     plt.show()  

#-----Notes:-----
# P1 is a copy of P2
#20230620_104 Has an issue with the head marker. It get detached from the head at the second 255