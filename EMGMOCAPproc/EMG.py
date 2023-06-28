import numpy as np


# def EMG_means(MOCAP_EMG,trial):
#        #for i, trial in enumerate (MOCAP_EMG):
#         EMG_means={"mean_full":dict(),"mean_flex":dict(),"mean_ext":dict()}
#         for key in MOCAP_EMG[trial]['TBound'].keys():
#             if not key == "is_set":
                
#                 time = np.arange(MOCAP_EMG[trial]['EMG'].shape[0])/2000
#                 ranges = MOCAP_EMG[trial]['TBound'][key]['ext'].reshape([-1,2])
#                 ranges2 = MOCAP_EMG[trial]['TBound'][key]['flex'].reshape([-1,2])
                
#                 cond_full = (np.logical_and(time > ranges2[:,:1] , time < ranges[:,1:2])).any(axis=0)
#                 cond_flex = (np.logical_and(time > ranges[:,:1] , time < ranges[:,1:2])).any(axis=0)
#                 cond_ext =(np.logical_and(time > ranges2[:,:1] , time < ranges2[:,1:2])).any(axis=0)
                
#                 means_full = MOCAP_EMG[trial]['EMG'][cond_full,:].mean(axis = 0)
#                 EMG_means["mean_full"].update({key:means_full})
                
#                 means_flex = MOCAP_EMG[trial]['EMG'][cond_flex,:].mean(axis = 0)   
#                 EMG_means["mean_flex"].update({key:means_flex})
                
#                 means_ext = MOCAP_EMG[trial]['EMG'][cond_ext,:].mean(axis = 0)
#                 EMG_means["mean_ext"].update({key:means_ext})
                
#                 #aux.update({"mean_full":means_full,"mean_flex":means_flex,"mean_ext":means_ext})
#                 #EMG_means.update({key:aux})
#         return EMG_means
#         #MOCAP_EMG[trial].update({"EMG_mean":EMG_means})

def EMG_means(MC_EMG,condition):     
        time = np.arange(MC_EMG.EMG.shape[1])/MC_EMG.sfemg
        EMGs = np.empty((2,0))
        for motion in MC_EMG.mot: 
            rflex = MC_EMG.Tbound[motion][:1,:].reshape([-1,2])
            rext = MC_EMG.Tbound[motion][1:,:].reshape([-1,2])
            
            if condition == "mean_full": 
                cond = (np.logical_and(time > rflex[:,:1] , time < rext[:,1:2])).any(axis=0)
                
            elif condition == "mean_flex":
                cond = (np.logical_and(time > rflex[:,:1] , time < rflex[:,1:2])).any(axis=0)
            
            elif condition == "mean_ext":
                cond =(np.logical_and(time > rext[:,:1] , time < rext[:,1:2])).any(axis=0)
            else:
                raise NameError("Non valid condition")
            EMG_select = MC_EMG.EMG[:,cond] 
            EMGs_stats = np.vstack(( EMG_select.mean(axis = 1) , EMG_select.std(axis=1)))
            EMGs=np.hstack((EMGs,EMGs_stats))
             
        return EMGs
