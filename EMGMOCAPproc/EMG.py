import numpy as np
from scipy.signal import butter, filtfilt , find_peaks
import os
import pandas
from ezc3d import c3d

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
    
def EMG_stats(trial , phase , channels=[0,8,9,10] , debug=False , Norm=[None] , metric="mean"):     
    """ 
    Function that get the EMG statistical value for a given phase in every labeled movement in the trial
    Parameters
    ----------
    trial : trial_MOCAP_EMG
        Object that contains the EMG and MOCAP data for a given condiiton for one participant
    phase : string
        portion of the movement being analized. Valid names: full_cycle, holding, approaching, recovery, 0_50, 20_70 
    channels : list
        DESCRIPTION. Analized channels. The default is [0,8,9,10].
    debug : TYPE, optional
        DESCRIPTION. The default is False.
    Norm : TYPE, optional
        DESCRIPTION. The default is [None].
    metric : TYPE, optional
        DESCRIPTION. The default is "mean".

    Raises
    ------
    NameError
        DESCRIPTION.

    Returns
    -------
    EMGs_stats_trial : 3D matrix [Nchans,Nmots,2]
        DESCRIPTION. matrix with statistical value (min/avg/max and std) of each channel for each motion

    """
        
    chan = np.array(channels, dtype=np.intp)
    EMG_chans = EMG_filt6( trial.EMG[chan]  , trial.sfemg)
    
    time = (np.arange(EMG_chans.shape[1])+1300)/trial.sfemg
    EMGs_stats_trial = np.empty((0,len(channels),2))
    
    for motion in trial.mot:
        try:
            mt=motion
            trial.Tbound[mt]
        except KeyError:
            print(motion)
            mt=motion[0]+str(np.sign(int(motion[1:]))*30)
        
        if trial.Tbound[mt].sum() == 0:
            print(trial.label)
            print(f"The limits are not set for {motion}")
            EMGs_stats_trial=np.vstack((EMGs_stats_trial,np.zeros((1,len(channels),2))))
            continue
        rflex = trial.Tbound[mt][:1,:].reshape([-1,2])
        rext = trial.Tbound[mt][1:,:].reshape([-1,2])
        
        if phase == "full_cicle": 
            cond = (np.logical_and(time > rflex[:,:1] , time < rext[:,1:2])).any(axis=0)
        elif phase == "0_50": 
            cond = (np.logical_and(time > rflex[:,:1] , time < (rext[:,:1])+rflex[:,1:2])/2).any(axis=0)
        elif phase == "holding": 
            cond = (np.logical_and(time > rflex[:,1:2] , time < rext[:,:1])).any(axis=0)    
        elif phase == "20_70":
            delta=rext[:,1:2]-rflex[:,:1] 
            cond = (np.logical_and(time > rflex[:,:1]+0.2*delta, time < rflex[:,:1]+0.8*delta)).any(axis=0)
        elif phase == "aproaching":
            cond = (np.logical_and(time > rflex[:,:1] , time < rflex[:,1:2])).any(axis=0)
            if rflex[0,1] > rflex[0,1]:
                print(trial.label)
        elif phase == "recovery":
            cond =(np.logical_and(time > rext[:,:1] , time < rext[:,1:2])).any(axis=0)
            if rext[0,1] > rext[0,1]:
                print(trial.label)
        else:
            raise NameError("Non valid phase: chose between full_cycle, holding, approaching, recovery, 0_50, 20_70")
        if debug:
            print(motion)
            print(f"from {min(time[cond])} to {max(time[cond])} ")
        
        if EMG_chans[:,cond].size == 0:
            print(EMG_chans.shape[1]+1300)
            print(trial.sfemg)
            print(trial.label)
            print(motion)
            print(rflex)
            print(rext)
            print(time[int(rflex[0,0]*2000):int(rflex[0,1]*2000)])
        EMG_phase_mot = EMG_chans[:,cond]

        if Norm[0]!= None:
            EMG_phase_mot/=Norm.reshape(-1,1)
        
        if metric=="mean":
            EMGs_stats_mot = np.dstack((EMG_phase_mot.mean(axis = 1) , EMG_phase_mot.std(axis=1)))
        elif metric=="max":
            EMGs_stats_mot = np.dstack((EMG_phase_mot.max(axis = 1) , EMG_phase_mot.std(axis=1)))
        elif metric=="min":
            EMGs_stats_mot = np.dstack((EMG_phase_mot.min(axis = 1) , EMG_phase_mot.std(axis=1)))
        else:
            print("The entered metric does not exist")
            break

        EMGs_stats_trial=np.vstack((EMGs_stats_trial,EMGs_stats_mot))
    return EMGs_stats_trial

def norm_EMG_part(part_lab,trials,channels,m_n,nMusc,crit="manual"):
    """
    Function that return representative normalization value for the set of trials of a given participant

    Parameters
    ----------
    part_lab : string
        participants name
    trials : list of trial_MOCAP_EMG objects
        EMG/MOCAP datastructure form each trial
    channels : int list
        channels that are included in the analysis
    m_n : int
        index of the analized muscle (from 0 to nMusc-1)
    nMusc : int
        amount of muscle measured in the study
    crit : string
        DESCRIPTION. Critiria to establish the normalization value:
            -manual: the values are normally added
            -mvc: values comes from the maximal volumetric contration test
            - <mot>;<phase>;<stat> the statistical value at the phase of a give motion is taken as a reference

    Returns
    -------
    None.

    """
    if crit =="manual":
        max_EMG_abs=np.array([1,1,1,1])
    elif crit =="Abs_peak":
        max_EMG_val_ind=np.array([max_EMG(trials[-i],channels) for i in range(1,3)])
    
        max_EMG_val=max_EMG_val_ind[:,0,:]
        max_EMG_ind=max_EMG_val_ind[:,1,:]
        max_EMG_abs=max_EMG_val.max(axis=0)
    
    elif crit =="MVC":
        mov=["F","E"]
        MEMG_means=np.zeros((2,nMusc))
        for n_m , m in enumerate(mov):
            file_address=os.path.join(".\\normEMG",f"{part_lab}_MVC_{m}.c3d")
            c = c3d(file_address)

            #Emg Data --------------------------------------------------------------
            analog_labels=c['parameters']["ANALOG"]["LABELS"]["value"]
            analog_idx= [analog_labels.index(f"Sensor {i}.EMG{i}") for i in range(1,1+nMusc)]
            analog_data = c['data']['analogs']
            MEMG_raw= analog_data[0,analog_idx,:]
            
            sfemg=c['header']['analogs']['frame_rate'] #Sampling frequency of EMG
            MEMG_f=EMG_filt3(MEMG_raw,sfemg)
            MEMG_means[n_m]=MEMG_f.mean(axis = 1)
        max_EMG_abs=MEMG_means.max(axis=0)
        
        
    else:
        # example: r;full_cicle;max
        rphase=crit.split(";")[-2]
        rmet=crit.split(";")[-1]
        ref_EMG=[]
        for trial in trials:
            abs_val=EMG_stats(trial,rphase, Norm=[None], channels=[channels[m_n]] , metric=rmet)[:,0,:]
            mot_ind=trial.mot.index("r15")
            ref_EMG.append(abs_val[mot_ind])
            mot_ind=trial.mot.index("r-15")
            ref_EMG.append(abs_val[mot_ind])
        max_EMG_abs = np.array(ref_EMG).max()
        max_EMG_abs = np.array([max_EMG_abs ] *4) # This is just  to get the same 4*1 format for the variable
        # check mean emg act during approaching
        #pick the max value
    return max_EMG_abs
    
def max_EMG(part,channels=[0,1,2,3],muscle=None):
    chan = np.array(channels, dtype=np.intp)
    EMGs = part.EMG[chan] 
    EMGs=EMG_filt2(EMGs,part.sfemg)
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
    print(peaks_chs)          
    if muscle == None:
        return (EMGs.max(axis=1) , EMGs.argmax(axis=1))
    else:
        return (EMGs[muscle].max(axis=0) , EMGs[muscle].argmax(axis=0))
    
def moving_average(a, n=3):
    ret = np.cumsum(a, axis=-1,dtype=float)
    ret[:,n:] = ret[:,n:] - ret[:,:-n]
    return ret[:,n - 1:] / n