import numpy as np
from scipy.signal import butter, filtfilt , find_peaks
import pandas
# def EMG_means(MOCAP_EMG,trial):
#        #for i, trial in enumerate (MOCAP_EMG):
#         EMG_means={"full_cicle":dict(),"flexion":dict(),"extension":dict()}
#         for key in MOCAP_EMG[trial]['TBound'].keys():
#             if not key == "is_set":
                
#                 time = np.arange(MOCAP_EMG[trial]['EMG'].shape[0])/2000
#                 ranges = MOCAP_EMG[trial]['TBound'][key]['ext'].reshape([-1,2])
#                 ranges2 = MOCAP_EMG[trial]['TBound'][key]['flex'].reshape([-1,2])
                
#                 cond_full = (np.logical_and(time > ranges2[:,:1] , time < ranges[:,1:2])).any(axis=0)
#                 cond_flex = (np.logical_and(time > ranges[:,:1] , time < ranges[:,1:2])).any(axis=0)
#                 cond_ext =(np.logical_and(time > ranges2[:,:1] , time < ranges2[:,1:2])).any(axis=0)
                
#                 means_full = MOCAP_EMG[trial]['EMG'][cond_full,:].mean(axis = 0)
#                 EMG_means["full_cicle"].update({key:means_full})
                
#                 means_flex = MOCAP_EMG[trial]['EMG'][cond_flex,:].mean(axis = 0)   
#                 EMG_means["flexion"].update({key:means_flex})
                
#                 means_ext = MOCAP_EMG[trial]['EMG'][cond_ext,:].mean(axis = 0)
#                 EMG_means["extension"].update({key:means_ext})
                
#                 #aux.update({"full_cicle":means_full,"flexion":means_flex,"extension":means_ext})
#                 #EMG_means.update({key:aux})
#         return EMG_means
#         #MOCAP_EMG[trial].update({"EMG_mean":EMG_means})

def EMG_filt(EMG_raw,Fs):  
    """
    Function that filter a multichannel EMG signal using a series filters in cascade

    Parameters
    ----------
    EMG_raw : np.array
        The original EMG signal
    Fs : int
        sampling frecuency in sample per second
    Returns
    -------
    EMG_f: np.array
         the filtered EMG signal

    """
    EMG_f=EMG_raw.copy()
    # Define the filter parameters
    order = 4  # Filter order
    cutoff_freqs = [[10,400],[30],[49,51],None]  # Cutoff frequencies for filtering
    ftypes= ["bandpass","highpass","bandstop","rectification"]

    if EMG_f.shape[1]==0:
        return np.zeros((EMG_f.shape[0],1))
    else:
        for cutoff_freq , bt in zip(cutoff_freqs,ftypes):
            # Design the Butterworth low-pass filter
            if bt =="DCrem":
                EMG_f -= EMG_f.mean(axis=1).reshape(-1,1) #DC component remotion
            elif bt == "rectification":
                EMG_f=abs(EMG_f)
            else:
                b, a = butter(order, cutoff_freq, btype=bt, analog=False, output='ba',fs=Fs)
                # Apply the filter to the curve
                EMG_f = filtfilt(b, a,EMG_f)
        return EMG_f
    
def EMG_filt2(EMG_raw,Fs):
    EMG_f=EMG_raw.copy()
    #EMG_f-=EMG_f.mean(axis=1).reshape(-1,1)
    # Select the desired array to filter and plot (e.g., index 0 for the first array)
    # Define the filter parameters
    order = 4  # Filter order
    #cutoff_freqs = [[30,400],[2],[2.5]]  # Cutoff frequencies for filtering
    #btypes= ["bandpass","highpass","lowpass"]
    cutoff_freqs = [[30,400]]  # Cutoff frequencies for filtering
    btypes= ["bandpass"]
    if EMG_f.shape[1]==0:
        return np.zeros((EMG_f.shape[0],1))
    else:
        for cutoff_freq , bt in zip(cutoff_freqs,btypes):
            # Design the Butterworth low-pass filter
            b, a = butter(order, cutoff_freq, btype=bt, analog=False, output='ba',fs=Fs)
            # Apply the filter to the curve
            EMG_f = filtfilt(b, a,EMG_f)
            if bt=="bandpass":
               EMG_f=abs(EMG_f)  
        EMG_f=moving_average(EMG_f, n=20) 
        return EMG_f

def EMG_filt3(EMG_raw,Fs):
    """
    Function that filter a multichannel EMG signal using a series filters in cascade

    Parameters
    ----------
    EMG_raw : np.array
        The original EMG signal
    Fs : int
        sampling frecuency in sample per second
    Returns
    -------
    EMG_f: np.array
         the filtered EMG signal

    """
    EMG_f=EMG_raw.copy()
    # Define the filter parameters
    order = 4  # Filter order
    cutoff_freqs = [None,[2],None,[2.5],None]  # Cutoff frequencies for filtering
    ftypes= ["DCrem","highpass","rectification","lowpass","MAvg"] # Type of filter

    if EMG_f.shape[1]==0:
        return np.zeros((EMG_f.shape[0],1))
    else:
        for cutoff_freq , bt in zip(cutoff_freqs,ftypes):
            # Design the Butterworth low-pass filter
            if bt =="DCrem":
                EMG_f -= EMG_f.mean(axis=1).reshape(-1,1) #DC component remotion
            elif bt == "rectification":
                EMG_f=abs(EMG_f)
            elif bt == "MAvg":
                wsize=1300
                EMG_f=moving_average(EMG_f, n=wsize) 
            else:
                b, a = butter(order, cutoff_freq, btype=bt, analog=False, output='ba',fs=Fs)
                # Apply the filter to the curve
                EMG_f = filtfilt(b, a,EMG_f)
        return EMG_f


def hampel_filter(data, window_size=7, n=3):
    """
    Apply Hampel filter to the input data.
    
    Parameters:
        data (array_like): Input time series data.
        window_size (int): Size of the sliding window.
        n_sigma (float): Number of standard deviations used to define the threshold.
    
    Returns:
        filtered_data (ndarray): Filtered data.
    """
    filtered_data = data
    orig_data= data.copy()
    len_dat=orig_data.shape[-1]
    print(len_dat)
    for i in range(len_dat):
        window = orig_data[:,max(0, i - window_size//2):min(len_dat, i + window_size//2 + 1)]
        median = np.median(window,axis=1).reshape(-1,1)
        deviation = np.abs(window - median)
        threshold = n * np.std(window,axis=1).reshape(-1,1)
        outliers = (deviation > threshold).any(axis=1)
        if outliers.any():
            print(i)
        filtered_data[outliers,i]=median[outliers,0]

    return filtered_data
 
    
    
def EMG_filt4(EMG_raw,Fs):
    """
    Function that filter a multichannel EMG signal using a series filters in cascade

    Parameters
    ----------
    EMG_raw : np.array
        The original EMG signal
    Fs : int
        sampling frecuency in sample per second
    Returns
    -------
    EMG_f: np.array
         the filtered EMG signal

    """
    EMG_f=EMG_raw.copy()
    # Define the filter parameters
    order = 4  # Filter order
    cutoff_freqs = [None,[30,400],[200,4],[6]]  # Cutoff frequencies for filtering
    ftypes= ["DCrem","bandpass","hampel","lowpass"]#,"MAvg"] # Type of filter

    if EMG_f.shape[1]==0:
        return np.zeros((EMG_f.shape[0],1))
    else:
        for cutoff_freq , bt in zip(cutoff_freqs,ftypes):
            # Design the Butterworth low-pass filter
            if bt =="DCrem":
                EMG_f -= EMG_f.mean(axis=1).reshape(-1,1) #DC component remotion
            elif bt == "rectification":
                EMG_f=abs(EMG_f)
            elif bt == "MAvg":
                wsize=1300
                EMG_f=moving_average(EMG_f, n=wsize)
            elif bt == "hampel":
                wsize=cutoff_freq[0]
                n=cutoff_freq[1]
                EMG_f=hampel_filter(EMG_f, window_size=wsize , n=n)
                
            else:
                b, a = butter(order, cutoff_freq, btype=bt, analog=False, output='ba',fs=Fs)
                # Apply the filter to the curve
                EMG_f = filtfilt(b, a,EMG_f)
        return EMG_f

def EMG_filt5(EMG_raw,Fs):
    """
    Function that filter a multichannel EMG signal using a series filters in cascade

    Parameters
    ----------
    EMG_raw : np.array
        The original EMG signal
    Fs : int
        sampling frecuency in sample per second
    Returns
    -------
    EMG_f: np.array
         the filtered EMG signal

    """
    EMG_f=EMG_raw.copy()
    # Define the filter parameters
    order = 4  # Filter order
    cutoff_freqs = [None,[2],None,[2.5]]  # Cutoff frequencies for filtering
    ftypes= ["DCrem","highpass","rectification","lowpass"] # Type of filter

    if EMG_f.shape[1]==0:
        return np.zeros((EMG_f.shape[0],1))
    else:
        for cutoff_freq , bt in zip(cutoff_freqs,ftypes):
            # Design the Butterworth low-pass filter
            if bt =="DCrem":
                EMG_f -= EMG_f.mean(axis=1).reshape(-1,1) #DC component remotion
            elif bt == "rectification":
                EMG_f=abs(EMG_f)              
            else:
                b, a = butter(order, cutoff_freq, btype=bt, analog=False, output='ba',fs=Fs)
                # Apply the filter to the curve
                EMG_f = filtfilt(b, a,EMG_f)
        return EMG_f
    
def EMG_filt6(EMG_raw,Fs):
    """
    Function that filter a multichannel EMG signal using a series filters in cascade THIS ONE WORKS

    Parameters
    ----------
    EMG_raw : np.array
        The original EMG signal
    Fs : int
        sampling frecuency in sample per second
    Returns
    -------
    EMG_f: np.array
         the filtered EMG signal

    """
    EMG_f=EMG_raw.copy()
    # Define the filter parameters
    order = 4  # Filter order
    cutoff_freqs = [None,[15,400],None,None]  # Cutoff frequencies for filtering ---------- pb 15 w 250
    ftypes= ["DCrem","bandpass","rectification","MAvg"] # Type of filter

    if EMG_f.shape[1]==0:
        return np.zeros((EMG_f.shape[0],1))
    else:
        for cutoff_freq , bt in zip(cutoff_freqs,ftypes):
            # Design the Butterworth low-pass filter
            if bt =="DCrem":
                EMG_f -= EMG_f.mean(axis=1).reshape(-1,1) #DC component remotion
            elif bt == "rectification":
                EMG_f=abs(EMG_f)
            elif bt == "MAvg":
                wsize=250
                EMG_f=moving_average(EMG_f, n=wsize)
          
            else:
                b, a = butter(order, cutoff_freq, btype=bt, analog=False, output='ba',fs=Fs)
                # Apply the filter to the curve
                EMG_f = filtfilt(b, a,EMG_f)
        return EMG_f
    
def EMG_filt6(EMG_raw,Fs):
    """
    Function that filter a multichannel EMG signal using a series filters in cascade THIS ONE WORKS

    Parameters
    ----------
    EMG_raw : np.array
        The original EMG signal
    Fs : int
        sampling frecuency in sample per second
    Returns
    -------
    EMG_f: np.array
         the filtered EMG signal

    """
    EMG_f=EMG_raw.copy()
    # Define the filter parameters
    order = 4  # Filter order
    cutoff_freqs = [None,[15,400],None,None]  # Cutoff frequencies for filtering ---------- pb 15 w 250
    ftypes= ["DCrem","bandpass","rectification","MAvg"] # Type of filter

    if EMG_f.shape[1]==0:
        return np.zeros((EMG_f.shape[0],1))
    else:
        for cutoff_freq , bt in zip(cutoff_freqs,ftypes):
            # Design the Butterworth low-pass filter
            if bt =="DCrem":
                EMG_f -= EMG_f.mean(axis=1).reshape(-1,1) #DC component remotion
            elif bt == "rectification":
                EMG_f=abs(EMG_f)
            elif bt == "MAvg":
                wsize=250
                EMG_f=moving_average(EMG_f, n=wsize)
          
            else:
                b, a = butter(order, cutoff_freq, btype=bt, analog=False, output='ba',fs=Fs)
                # Apply the filter to the curve
                EMG_f = filtfilt(b, a,EMG_f)
        return EMG_f
    
def EMG_stats(MC_EMG , condition , channels=[0,8,9,10] , debug=False , Norm=[None] , metric="mean"):     
        
        
        chan = np.array(channels, dtype=np.intp)
        EMG_chans = EMG_filt6( MC_EMG.EMG[chan]  , MC_EMG.sfemg)
        
        time = (np.arange(EMG_chans.shape[1])+1300)/MC_EMG.sfemg
        EMGs = np.empty((0,len(channels),2))
        
        for motion in MC_EMG.mot:
            try:
                mt=motion
                MC_EMG.Tbound[mt]
            except KeyError:
                print(motion)
                mt=motion[0]+str(np.sign(int(motion[1:]))*30)
            
            if MC_EMG.Tbound[mt].sum() == 0:
                print(MC_EMG.label)
                print(f"The limits are not set for {motion}")
                EMGs=np.vstack((EMGs,np.zeros((1,len(channels),2))))
                continue
            rflex = MC_EMG.Tbound[mt][:1,:].reshape([-1,2])
            rext = MC_EMG.Tbound[mt][1:,:].reshape([-1,2])
            if condition == "full_cicle": 
                cond = (np.logical_and(time > rflex[:,:1] , time < rext[:,1:2])).any(axis=0)
            elif condition == "first_half": 
                cond = (np.logical_and(time > rflex[:,:1] , time < (rext[:,:1])+rflex[:,1:2])/2).any(axis=0)
            elif condition == "holding": 
                cond = (np.logical_and(time > rflex[:,1:2] , time < rext[:,:1])).any(axis=0)    
            elif condition == "20_70":
                delta=rext[:,1:2]-rflex[:,:1] 
                cond = (np.logical_and(time > rflex[:,:1]+0.2*delta, time < rflex[:,:1]+0.8*delta)).any(axis=0)
            elif condition == "aproaching":
                cond = (np.logical_and(time > rflex[:,:1] , time < rflex[:,1:2])).any(axis=0)
                if rflex[0,1] > rflex[0,1]:
                    print(MC_EMG.label)
            elif condition == "recovery":
                cond =(np.logical_and(time > rext[:,:1] , time < rext[:,1:2])).any(axis=0)
                if rext[0,1] > rext[0,1]:
                    print(MC_EMG.label)
            else:
                raise NameError("Non valid condition: chose between mean_full, flexion or extension")
            if debug:
                print(motion)
                print(f"from {min(time[cond])} to {max(time[cond])} ")
            
            if EMG_chans[:,cond].size == 0:
                print(EMG_chans.shape[1]+1300)
                print(MC_EMG.sfemg)
                print(MC_EMG.label)
                print(motion)
                print(rflex)
                print(rext)
                print(time[int(rflex[0,0]*2000):int(rflex[0,1]*2000)])
            EMG_select = EMG_chans[:,cond]

            #print(EMG_select.shape)
            #EMG_select=moving_average(EMG_select,300)
            if Norm[0]!= None:
                #EMG_select/=EMG_select.max(axis=1).reshape(-1,1)
                EMG_select/=Norm.reshape(-1,1)
            #EMG_select-=EMG_select.min(axis=-1).reshape(-1,1)
            #print(EMG_select.min(axis=-1))
            #print(EMG_select.min(axis=-1))
            #EMGs_stats = np.dstack((EMG_select.mean(axis = 1) , EMG_select.std(axis=1)))
            
            if metric=="mean":
                EMGs_stats = np.dstack((EMG_select.mean(axis = 1) , EMG_select.std(axis=1)))
            elif metric=="max":
                EMGs_stats = np.dstack((EMG_select.max(axis = 1) , EMG_select.std(axis=1)))
            elif metric=="min":
                EMGs_stats = np.dstack((EMG_select.min(axis = 1) , EMG_select.min(axis=1)))
            else:
                print("The entered metric does not exist")
                break
            #print(max(abs(EMG_select.std(axis=1))))
            EMGs=np.vstack((EMGs,EMGs_stats))
        return EMGs
    
def max_EMG(part_HS2,channels=[0,1,2,3],muscle=None):
    chan = np.array(channels, dtype=np.intp)
    EMGs = part_HS2.EMG[chan] 
    EMGs=EMG_filt2(EMGs,part_HS2.sfemg)
    #EMGs=
    if muscle == None:
        return (EMGs.max(axis=1) , EMGs.argmax(axis=1))
    else:
        return (EMGs[muscle].max(axis=0) , EMGs[muscle].argmax(axis=0))

def max_EMG(part_HS2,channels=[0,1,2,3],muscle=None):
    chan = np.array(channels, dtype=np.intp)
    EMGs = part_HS2.EMG[chan] 
    EMGs=EMG_filt2(EMGs,part_HS2.sfemg)
    #EMGs=
    if muscle == None:
        return (EMGs.max(axis=1) , EMGs.argmax(axis=1))
    else:
        return (EMGs[muscle].max(axis=0) , EMGs[muscle].argmax(axis=0))

def max_EMG_peaks(part_HS2,channels=[0,1,2,3],muscle=None):
    chan = np.array(channels, dtype=np.intp)
    EMGs = part_HS2.EMG[chan] 
    EMGs=EMG_filt2(EMGs,part_HS2.sfemg)
    peaks_chs=[find_peaks(EMG,height=0.01*max(EMG),width=2000) for EMG in EMGs]
    # peaks_abs= np.zeros
    # for i, peaks_1ch in enumerate(peaks_chs):
    #     mpeak=EMG[i,peaks_1ch].argmax()
    #     for peak in peaks_1ch[1:]:
    #         if peak[i,peak]>0.2*peak[i,peak]:
                
    print(peaks_chs)            
                
                
    #EMGs=
    if muscle == None:
        return (EMGs.max(axis=1) , EMGs.argmax(axis=1))
    else:
        return (EMGs[muscle].max(axis=0) , EMGs[muscle].argmax(axis=0))
    
def moving_average(a, n=3):
    ret = np.cumsum(a, axis=-1,dtype=float)
    ret[:,n:] = ret[:,n:] - ret[:,:-n]
    return ret[:,n - 1:] / n