from ezc3d import c3d
import numpy as np
import matplotlib.pyplot as plt
import quaternion
import pickle

file_address = r'G:\My Drive\PhD\My research\EMG and MOCAP measurements\4- 6 bar rigid blockage, single user\2023060907.c3d'


def crd2dict (file_address):
    c = c3d(file_address)

    #MoCap Points -----------------------------------------------------------
    point_data = c['data']['points'][0:3,:,:]

    marker_dict=dict()

    used_markers= 4

    for i , marker_name in enumerate(c['parameters']['POINT']['LABELS']["value"]):
        marker_dict.update({marker_name:point_data[0:3,i,:]}) 
        if i == used_markers:
            break

    #Emg Data --------------------------------------------------------------
    analog_data = c['data']['analogs']

    EMG_dict = {"EMG":  analog_data[0,:,:]}

    Global_dict = {'MOCAP':marker_dict, 'EMG':EMG_dict }

    return Global_dict


def GetQuat(a,b):
    rotAxis = np.cross(a , b);
    rotAxisNorm = np.linalg.norm(rotAxis);
    aNorm = np.linalg.norm(a);
    bNorm = np.linalg.norm(b);
    ang = np.arcsin(rotAxisNorm/(aNorm*bNorm));
    rotAxis = rotAxis / rotAxisNorm * ang;  
    
    return quaternion.from_rotation_vector(rotAxis)

def get_kin_MOCAP(dirs,matfolder):
    """
    Function that takes converted converted c3d data in mat file and return dict with angles and
    EMG recording
    
    Input: 
        .dirs: absolute directories of .mat files

    Output:
        .Q_MOCAP: dictionary with the orientation and EMG data for each case
    """
    
    Q_MOCAP =dict.fromkeys(dirs) 
    for file in Q_MOCAP.keys():
        Q_MOCAP[file]={"RPY":None , "EMG": None , "Mkr_samp": dict.fromkeys(["head","back"])}
        
        Tdict = dict.fromkeys(["s","ll","lr","rl","rr"])
        [Tdict.update({key:dict.fromkeys(["flex","ext"],None)}) for key in Tdict.keys()]
        
        Tdict.update({"is_set": False})
        Q_MOCAP[file].update({"TBound":Tdict})
        
        "Kinematic analysis"
            
        
        fullMatFileName = matfolder +  file;

        with open(fullMatFileName, 'rb') as f:
            s = pickle.load(f)
        
        HeadTraj = {'frontl' : s['MOCAP']['hlf'] , 'backl' : s['MOCAP']['hlb'] , 'frontr': s['MOCAP']['hrf'] , 'backr': s['MOCAP']['hrb']}; 
        ShouldTraj = {'SupL': s['MOCAP']['dlt'] , 'SupR': s['MOCAP']['drt'] , 'InfL' : s['MOCAP']['dlb'] , 'InfR': s['MOCAP']['drb']}; 
        TopBarTraj = {'Cent': s['MOCAP']['d1c'] , 'Right': s['MOCAP']['d1r'] , 'Left': s['MOCAP']['d1l'] }
        
        globalFrame = np.eye(3); 
        
        Should_orient = get_orient(ShouldTraj , globalFrame , axis_form_square_arrangement_back);
        Head_orient  = get_orient(HeadTraj , globalFrame , axis_form_square_arrangement_head);
                
        #Head_orient = Head_orient         
        #Should_orient = np.full(Should_orient.shape,quaternion.quaternion(1,0,0,0))
            
        headYPR_abs = getYPR(Head_orient)
        bodyYPR_abs = getYPR(Should_orient)

        Q_MOCAP[file].update({"RPY": np.array(headYPR_abs)})
        
        Q_MOCAP[file]["EMG"] = s["D"]["Devices"]["datalp"]
        Q_MOCAP[file]["Mkr_samp"]["head"] = s['MOCAP']['hlf']
        Q_MOCAP[file]["Mkr_samp"]["back"] = s['MOCAP']['dlt']
        
        
    return Q_MOCAP  


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
                             Example: RefLocal[i,:,:]=[[ux,vx,wx],[uy,vy,wy],[uz,vz,wz]] 
                         
     Output:
         Orient: quaternion array [samples x ] with orientation of the body over time
    """
     
     Orient=MrkConf(Markers)
     
     Orient= quaternion.from_rotation_matrix(Orient) # array of quaternions [samples, Qdim]
     
     
     Orient= Orient * (Orient[0].conjugate()); # Referering the head frames to the initial frame
     # Observation: because the variable frame is a numpy array it allow broadcasting with element wise products. Element-wise operations are ufunc with this property. 
     # Broadcasting rules: https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules
     
     ''' The orientation of both frames are aligned tih the global frame at the
     beginning of the experiment'''     
     return Orient

def axis_form_romboid_arrangement(Markers):
     """
     This function makes an ortonormal base from four markers in cross
     arrangement as shown in Figure 2 of [1], the normal vector in the z direction is used for determining the orientation
     Input:
         .Markers: struct with the fields: front, back, left, right 
         Each field is a [sample x 3] matrix with each marker temporal evolution

        . Output:
            RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                Example: RefLocal[i,:,:]=[[ux,vx,wx],[uy,vy,wy],[uz,vz,wz]] 
     """    
     u = Markers['front'] - Markers['back'];
     unorm = np.linalg.norm(u,axis=1).reshape(-1,1)
     u = u / unorm;
     
     
     
     v = Markers['left'] - Markers['right'];
     proj = np.sum(v*u,axis=1).reshape(-1,1);         #projection of v in u or viceversa
     v = (v- proj * u);                 # removing V component in the u direction
     vnorm = np.linalg.norm(v,axis=1).reshape(-1,1)
     v = v / vnorm;
     
     
     
     w = np.cross(u,v);
     wnorm = np.linalg.norm(w,axis=1).reshape(-1,1)
     w = w / wnorm; 
     
     
     u=u.reshape(-1,1,3) 
     v=v.reshape(-1,1,3) 
     w=w.reshape(-1,1,3)      
     RefLocal= np.hstack((u,v,w));               # Transformation matrix array [samples , xyz glob(3) , xyz loc(3), ]
     RefLocal= np.swapaxes(RefLocal,1,2)
     return RefLocal

def axis_form_square_arrangement_head(Markers):
     """
     This function makes an ortonormal base from four markers in the square arrangement like in the RoboSoft experiment
     Input:
         .Markers: struct with the fields: frontl, backl, leftr, backr 
         Each field is a [sample x 3] matrix with each marker temporal evolution

        . Output:
            RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                Example: RefLocal[i,:,:]=[[ux,vx,wx],[uy,vy,wy],[uz,vz,wz]] 
     """    
     u = (Markers['frontl'] - Markers['backl'])+(Markers['frontr'] - Markers['backr'])
     unorm = np.linalg.norm(u,axis=1).reshape(-1,1)
     u = u / unorm;
     
     
     
     v = (Markers['frontl'] - Markers['frontr'])+(Markers['backl'] - Markers['backr'])
     proj = np.sum(v*u,axis=1).reshape(-1,1);         #projection of v in u or viceversa
     v = (v- proj * u);                 # removing V component in the u direction
     vnorm = np.linalg.norm(v,axis=1).reshape(-1,1)
     v = v / vnorm;
     
     
     
     w = np.cross(u,v);
     wnorm = np.linalg.norm(w,axis=1).reshape(-1,1)
     w = w / wnorm; 
     
     
     u=u.reshape(-1,1,3) 
     v=v.reshape(-1,1,3) 
     w=w.reshape(-1,1,3)      
     RefLocal= np.hstack((u,v,w));               # Transformation matrix array [samples , xyz glob(3) , xyz loc(3), ]
     RefLocal= np.swapaxes(RefLocal,1,2)
     return RefLocal
 
def axis_form_square_arrangement_back (Markers):
     """
     This function makes an ortonormal base from four markers in the square arrangement like in the RoboSoft experiment
     Input:
         .Markers: struct with the fields: SupL, Supr, InfR, InfR 
         Each field is a [sample x 3] matrix with each marker temporal evolution

        . Output:
            RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                Example: RefLocal[i,:,:]=[[ux,vx,wx],[uy,vy,wy],[uz,vz,wz]] 
     """    
     w = (Markers['SupL'] - Markers['InfL'])+(Markers['SupR'] - Markers['InfR'])
     wnorm = np.linalg.norm(w,axis=1).reshape(-1,1)
     w = -w / wnorm;
     
     
     
     v = (Markers['SupL'] - Markers['SupR'])+(Markers['InfR'] - Markers['InfL'])
     proj = np.sum(v*w,axis=1).reshape(-1,1);         #projection of v in u or viceversa
     v = (v- proj * w);                 # removing V component in the u direction
     vnorm = np.linalg.norm(v,axis=1).reshape(-1,1)
     v = v / vnorm;
     
     
     
     u = np.cross(v,w);
     unorm = np.linalg.norm(u,axis=1).reshape(-1,1)
     u = u / unorm; 
     
     
     u=u.reshape(-1,1,3) 
     v=v.reshape(-1,1,3) 
     w=w.reshape(-1,1,3)      
     RefLocal= np.hstack((u,v,w));               # Transformation matrix array [samples , xyz glob(3) , xyz loc(3), ]
     RefLocal= np.swapaxes(RefLocal,1,2)
     return RefLocal
     

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

