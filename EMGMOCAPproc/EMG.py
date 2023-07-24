import numpy as np
from scipy.signal import butter, filtfilt

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

def EMG_filt(EMG_raw,Fs,rect=False):
    EMG_filt=EMG_raw.copy()
    # Select the desired array to filter and plot (e.g., index 0 for the first array)
    # Define the filter parameters
    order = 4  # Filter order
    cutoff_freqs = [[10,400],[30],[49,51]]  # Cutoff frequencies for filtering
    btypes= ["bandpass","highpass","bandstop"]
    
    
    for cutoff_freq , bt in zip(cutoff_freqs,btypes):
        # Design the Butterworth low-pass filter
        b, a = butter(order, cutoff_freq, btype=bt, analog=False, output='ba',fs=Fs)
        # Apply the filter to the curve
        EMG_filt = filtfilt(b, a,EMG_filt)
    if rect == True:
        EMG_rect=abs(EMG_filt)
        #print(EMG_rect.mean(axis=-1).reshape(-1,1))
        return EMG_rect
    else:
        return EMG_filt


def EMG_means(MC_EMG,condition , channels=[0,8,9,10]):     
        time = np.arange(MC_EMG.EMG.shape[1])/MC_EMG.sfemg
        EMGs = np.empty((0,len(channels),2))
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
                raise NameError("Non valid condition: chose between mean_full, mean_flex or mean_ext")
            chan = np.array(channels, dtype=np.intp)
            EMG_select = MC_EMG.EMG[chan]
            EMG_select = EMG_filt( EMG_select[:,cond] , MC_EMG.sfemg , rect=True)
            EMG_select=moving_average(EMG_select,300)
            #EMG_select-=EMG_select.min(axis=-1).reshape(-1,1)
            #print(EMG_select.min(axis=-1))
            #print(EMG_select.min(axis=-1))
            EMGs_stats = np.dstack((EMG_select.mean(axis = 1) , EMG_select.std(axis=1)))
            #print(max(abs(EMG_select.std(axis=1))))
            EMGs=np.vstack((EMGs,EMGs_stats))
            
             
        return EMGs
    
def moving_average(a, n=3):
    ret = np.cumsum(a, axis=-1,dtype=float)
    ret[:,n:] = ret[:,n:] - ret[:,:-n]
    return ret[:,n - 1:] / n