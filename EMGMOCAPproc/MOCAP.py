from ezc3d import c3d
import numpy as np
import matplotlib.pyplot as plt
import quaternion
import os
from scipy.signal import butter, filtfilt
import pandas

import sys 
EMGMOCAPpt =r"C:\Users\UTAIL\OneDrive\Documents\GitHub\MOCAP-EMG-proc"
if not sys.path.count(EMGMOCAPpt):
    sys.path.append(EMGMOCAPpt)
    
from EMGMOCAPproc import MRKs2REFs , AUX

# =============================================================================
# Auxiliar variables
# =============================================================================

DEF_HMkrs = MRKs2REFs.axis_form_square_arrangement_head
DEF_SMkrs = MRKs2REFs.axis_form_square_arrangement_back

# =============================================================================
# Functions
# =============================================================================
def crd2dict (file_addresses):
    
    c3ds=[]
    for file_address in file_addresses:
        
        print("File {} is being processed".format(os.path.split(file_address)[-1]))
        c = c3d(file_address)
    
        #MoCap Points -----------------------------------------------------------
        point_data = c['data']['points'][0:3,:,:]
    
        marker_dict=dict()
    
        used_markers= 15
    
        for i , marker_name in enumerate(c['parameters']['POINT']['LABELS']["value"]):
            mrk_name_wo_sub = marker_name.split(":")[-1]
            marker_dict.update({mrk_name_wo_sub: point_data[0:3,i,:].T}) 
            if i == used_markers:
                break 

        #Emg Data --------------------------------------------------------------
        analog_labels=c['parameters']["ANALOG"]["LABELS"]["value"]
        analog_idx= [analog_labels.index(f"Sensor {i}.EMG{i}") for i in range(1,7)]
        analog_data = c['data']['analogs']
    
        
        EMG_data = analog_data[0,analog_idx,:]
        #print(analog_data)
        #Sampling _frequency-----------------------------------------------------------------
        
        sfmc=c['header']['points']['frame_rate'] #Sampling frequency of MoCap
        sfemg=c['header']['analogs']['frame_rate'] #Sampling frequency of EMG
        
        # Making list of c3ds
        c3ds.append({'MOCAP':marker_dict, 'EMG':EMG_data ,"SFMC": sfmc , "SFEMG": sfemg})
    
    c3dfin=c3ds.pop(0)
    
    #print(c3dfin["MOCAP"].keys())
    for C3D in c3ds:
        c3dfin["MOCAP"]=np.append(c3dfin["MOCAP"],C3D["MOCAP"],axis=0)
        c3dfin["EMG"]=np.append(c3dfin["EMG"],C3D["EMG"],axis=1)
    return c3dfin

# =============================================================================
# Classes
# =============================================================================

class trial_MOCAP_EMG:
    """ Class that contains the kinematic and EMG data of one trial for a given participant
    """
    def __init__(self,c3ds, IMU= None,cond="",hmrks = DEF_HMkrs , smrks=DEF_SMkrs):
        """
        The class is created based on one or several concatenated crd files
        
        Input: 
            .c3ds: absolute directories of .c3d files that will be concatenated
            .cond: condition or trial number
        """
        self.Tbound = None
        self.mot = None
        self.label = cond
        
        if not IMU == None:
            with open(IMU,"r"):
                df = pandas.read_csv(IMU)
                self.IMU ={"meas": df.iloc[:,:3].to_numpy() , "target": df.iloc[:,3:].to_numpy() } 
                
        else:
            self.IMU = None
        
        "Kinematic analysis"
        
        s = crd2dict (c3ds)
        
        
        # HeadTraj = {'frontl' : s['MOCAP']['hlf'] , 'backl' : s['MOCAP']['hlb'] , 'frontr': s['MOCAP']['hrf'] , 'backr': s['MOCAP']['hrb']}; 
        # ShouldTraj = {'SupL': s['MOCAP']['dlt'] , 'SupR': s['MOCAP']['drt'] , 'InfL' : s['MOCAP']['dlb'] , 'InfR': s['MOCAP']['drb']}; 
        
        
        globalFrame = np.eye(3); 
        
        # Should_orient = get_orient(ShouldTraj , globalFrame , smrks);
        # Head_orient  = get_orient(HeadTraj , globalFrame , hmrks);
        
        Should_orient = get_orient(s['MOCAP'], globalFrame , smrks);
        Head_orient  = get_orient(s['MOCAP'], globalFrame , hmrks);
        
                
        #Head_orient = Head_orient         
        #Should_orient = np.full(Should_orient.shape,quaternion.quaternion(1,0,0,0))
            
        headYPR_abs = getYPR(Head_orient*Head_orient[0].conjugate())
        bodyYPR_abs = getYPR(Should_orient[0].conjugate()*Should_orient)
        
        headYPR_rel = getYPR((Head_orient*Should_orient.conjugate())*(Head_orient[0]*Should_orient[0].conjugate()).conjugate())
        
        self.q = {"head":Head_orient , "body": Should_orient}
        self.RPY = {"head_abs" : np.array(headYPR_abs) , "body_abs": np.array(bodyYPR_abs) , "head_rel" : np.array(headYPR_rel)}
        self.EMG= s["EMG"]
        self.Mrk_samp=dict()
        #self.Mrk_samp = {"head":s['MOCAP']['hlf'],"back":s['MOCAP']['dlt']}
        self.markers= s['MOCAP']
        self.sfmc = s["SFMC"]
        self.sfemg = s ["SFEMG"]
        
    def set_boundaries(self , seq,t_phases=np.empty((0,1))):
        """
        This function ask for manually indicate the boundary points of the flexion
        and extension of the motion sequence. Such a data is stored in the dictionary.
        The clicks are from left to right starting from the beginnning and end of each neck motion
        

        Input:
        dict_data : dictionary that contains the MOCAP data
        seq : list discribing which type of motion correspond each pair of values
            "s" is for sagital motion , "c" for coronal motion and "r" for axial rotation

        """
        
        Xl = r'time' 
        Yl= [r"$\alpha_{sag}$" , r"$\beta_{cor}$" , r"$\gamma_{ax}$"]         
        fig = plt.figure()
        ax = plt.axes()
        
        RPY = self.RPY["head_abs"]
        time= np.arange(RPY.shape[1])/self.sfmc
        ax.plot(time,RPY.T)
        ax.set_title(r"{:s}: ".format(self.label)) 
        ax.set_xlabel(Xl)
        #plt.ylabel(Yl_2)
        ax.legend(Yl)
        mng=plt.get_current_fig_manager()
        mng.full_screen_toggle()
        
        self.mot = list(set(seq))
        self.Tbound= dict.fromkeys(self.mot , np.array(np.empty((2,0))))

          
        nclicks = np.array(seq).size*4
        print("File {} . Make {} clicks".format(self.label,nclicks))
        #boundaries=plt.ginput(nclicks,timeout=-1)
        #boundaries=[bound[0] for bound in boundaries]
        if len(t_phases):
            #t_phases=t_phases[0]
            time_phases=np.cumsum(t_phases,axis=-1) 
            print(time_phases)
            phase_strt=(np.arange(len(seq))*time_phases[0,-1]).reshape(-1,1)
            time_phase_rel=np.hstack((np.zeros((1,1)),time_phases[:,:-1]))
            boundaries=(phase_strt+(time_phase_rel)).flatten()
        else:
            boundaries=AUX.get_bound(nclicks,fig,ax,seq)
            
        boundaries=np.array(boundaries).reshape(-1,2,2)
        for i , motion in enumerate(seq):            
            self.Tbound[motion] = np.append(self.Tbound[motion],boundaries[i,:,:],axis=1)
        plt.close(fig)
        
    def RPY_filtered(self,cutoff_freq=0.003):
        order = 4
        b, a = butter(order, cutoff_freq, btype='low', analog=False, output='ba')
        # Apply the filter to the curve
        # Calculate the vertical shift value
        filt_data=dict()
        for raw_angles in self.RPY.items():
            filt_data.update({raw_angles[0]:filtfilt(b, a, raw_angles[1])})
        return filt_data
        
    def seq_from_IMU(self,start=700,stride=2000):
        """
        Return sequence list from the recording of the GUI's recording of the target
        """
        reference=set(['s-13', 'l13', 'r-29', 'r9', 'r-9', 's40', 'l-40', 'l-13', 'l40', 's-40','r29', 's13'])
        target = self.IMU["target"]
        probe = min(np.argmin(target==0,axis=0))+start
        
        motions = ["s","r","l"]
        seq=[]
        while probe < target.shape[0]:
            ang_probe = target[probe,:]
            
            index = np.argmin(-abs(ang_probe))
            if abs(int(ang_probe[index]*180/np.pi)) == 30:
                ang_probe[index]=np.sign(ang_probe[index])*29*np.pi/180
            motion = motions [index] 
            seq.append(f"{motion}{int(ang_probe[index]*180/np.pi)}")
            probe += stride
        print(len(seq))
        if len(seq) == 11: 
            print(set(reference).difference(seq))
            seq.append(set(reference).difference(seq).pop())
        return seq
    
    def get_mean_ang(self, neutral = True, Ref="head_rel"):
        """ Returns mean angle in the roll yaw pich decomposition on a given range:
            Neutral TRUE -> Range is out of the displacement period
            Neutral FALSE -> Range is during the holding phase
            """
            
        if self.Tbound == None:
            print ("Define limits first")
            return
        time = np.arange(self.RPY[Ref].shape[1])/self.sfmc
        
        if neutral:
            flex_str= np.empty([0,1])
            ext_end= np.empty([0,1])
            
            for motion in self.mot: 
                rflex = self.Tbound[motion][:1,:].reshape([-1,2])[:,0:1]
                rext = self.Tbound[motion][1:,:].reshape([-1,2])[:,1:]
                
                ext_end=np.vstack((ext_end,rext))
                flex_str=np.vstack((flex_str,rflex))
            
            #sorting values
            ext_end.sort(axis=0)
            flex_str.sort(axis=0)
            ext_end = ext_end[:-1,:] #popping last value
            flex_str = flex_str[1:,:]  # popping first value
            
            #print(np.hstack((ext_end,flex_str)))    
            cond = (np.logical_and(time < flex_str , time > ext_end)).any(axis=0)    
            offset= np.vstack((self.RPY[Ref][:,cond].mean(axis = 1),self.RPY[Ref][:,cond].std(axis = 1)))
            
            return {"value": offset,"rng":np.hstack((ext_end,flex_str))}
            
        else:

            mean_ang = dict()
            for motion in self.mot:
                flex_end = self.Tbound[motion][:1,:].reshape([-1,2])[:,1:]
                ext_str = self.Tbound[motion][1:,:].reshape([-1,2])[:,0:1]               
                
                cond = (np.logical_and(time > flex_end , time < ext_str)).any(axis=0)
                    
                
                RPY_select = self.RPY[Ref][:,cond]
                RPY_stats = np.vstack(( RPY_select.mean(axis = 1) , RPY_select.std(axis=1)))
                mean_ang.update({motion:{"value":RPY_stats,"rng": np.hstack((flex_end,ext_str))}})
        return mean_ang
        
    
    def show_DSET_map():
        pardir = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
        imdir = os.path.join(pardir,"EXT","Q_MOCAP_Structure.png")
        fig, ax = plt.subplots()
            
        image = plt.imread(imdir)    
        
        ax.imshow(image)
        ax.axis('off')    
    
    def __str__(self):
         return f"{self.label}"
     

def get_orient(Markers,globalFrame,MrkConf):
     """
     This function returns the quaternion of a given body with a set of attached markers
     Input:
         .Markers: struct whose fields contain the temporal-spatial data from each marker 
         Each field is a [sample x 3] matrix with the temporal evolution of each marker
         . MrkConf: function which indicates the spatial configuration of the markers
             . Input: Markers
             . Output:
                 RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                             Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
                         
     Output:
         Orient: quaternion array [samples x ] with orientation of the body over time
    """
     
     Orient=MrkConf(Markers)
     
     Orient= quaternion.from_rotation_matrix(Orient) # array of quaternions [samples, Qdim]
     
     
     #Orient=Orient * Orient[0].conjugate(); # Referering the head frames to the initial frame
     # Observation: because the variable frame is a numpy array it allow broadcasting with element wise products. Element-wise operations are ufunc with this property. 
     # Broadcasting rules: https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules
     
     ''' The orientation of both frames are aligned tih the global frame at the
     beginning of the experiment'''     
     return Orient
     

getYPR = np.vectorize(lambda q: yawPitchRoll(q,ls=False),otypes=["f"]*3)



def yawPitchRoll(q, ls = False):
    """Function that converts quaternion to the yaw pitch roll representation
    This representation corresponds to a tait-bryan rotation of xyz-order.
    Input: 
        .q: quaternion
    Output:
        yaw , pitch roll in radians"""

    yaw = np.arctan2(2.0*(q.y*q.z + q.w*q.x), q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z);
    pitch = np.arcsin(-2.0*(q.x*q.z - q.w*q.y));
    roll = np.arctan2(2.0*(q.x*q.y + q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z);
    if ls:
        return [yaw , pitch , roll]
    else:
            arr = ((np.array((yaw , pitch , roll),dtype = 'float64')+np.pi)%(2*np.pi)-np.pi) * 180/np.pi
            return arr [0] , arr[1] , arr[2]

def swng_swch_dec(q,v):
    """Function that decompose quaternion into the swing-twist represantation (swing happens first) 
    Input: 
        .q: quaternion
        .v: base vector. I think it is the swing axis.
    Output:
        qs , qt : the quaternion for the swing and twist representation"""
    if np.linalg.norm(v)==0:
        raise TypeError("Null base vector for swing-switch representation ")
    q2spin = lambda  q: np.array((q.w,q.x,q.y,q.z)) 
    spin2q = lambda sp: quaternion.quaternion(sp.w,sp)
    
    x,y,z=v
    u=np.dot(q.vec,v)
    n=np.linalg.norm(v)
    m=q.w*n
    l=(m**2+u**2*n)**0.5
    qt=quaternion.quaternion(m/l,z*u/l,x*u/l,y*u/l)
    qs=q/qt
    return  qs ,qt

def swng_swch_dec_spn(q,v):
    """(NOT WORKING YET)Function that decompose quaternion into the swing-twist represantation (swing happens first) 
    Input: 
        .q: quaternion
        .v: base vector. I think it is the swing axis.
    Output:
        qs , qt : the quaternion for the swing and twist representation
    NOTE"""
    raise TypeError("FIX spinor issue")
    if np.linalg.norm(v)==0:
        raise TypeError("Null base vector for swing-switch representation ")
    q2spin = lambda  q: np.array((q.w,q.x,q.y,q.z)) 
    spin2q = lambda sp: quaternion.quaternion(sp.w,sp)
    
    x,y,z=v
    u=np.dot(q.vec,v)
    n=np.linalg.norm(v)
    m=q.w*n
    l=(m**2+u**2*n)**0.5
    qt=quaternion.quaternion(m/l,z*u/l,x*u/l,y*u/l)
    qs=q/qt
    return  qs ,qt

def swng_swch_dec_drct(q,v):
    """(NOT WORKING YET)Function that decompose quaternion into the twist after swing representation
    Input: 
        .q: quaternion
        .v: base vector. It is a generic vector that is normal to the sTring axis
        For verification: v=qs**-1*qt.vec*qs is the twist vector in the global frame
    Output:
        qs , qt : the quaternion for the swing and twist representation
    NOTE"""
    if np.linalg.norm(v)==0:
        raise TypeError("Null base vector for swing-switch representation ")
    qv=quaternion.from_vector_part(v) 
    

    w = (q*qv*1/q).vec
    n = np.cross(v,w)
    
    n/=np.linalg.norm(n)
    print(n)
    ca=np.dot(v,w)/np.linalg.norm(v)/np.linalg.norm(w)
    a=np.arccos(ca)
    ca2=np.cos(a/2)
    sa2=np.sin(a/2)
    
    qs=quaternion.quaternion(ca2,*(sa2*n))
    
    qt=q/qs
    
    return  qs ,qt