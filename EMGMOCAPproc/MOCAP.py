from ezc3d import c3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import quaternion
import pickle
import os

def crd2dict (file_address):
    c = c3d(file_address)

    #MoCap Points -----------------------------------------------------------
    point_data = c['data']['points'][0:3,:,:]

    marker_dict=dict()

    used_markers= 7

    for i , marker_name in enumerate(c['parameters']['POINT']['LABELS']["value"]):
             
        marker_dict.update({marker_name:point_data[0:3,i,:].T}) 
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

def get_kin_MOCAP(dirs,c3dfolder):
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
        
        Tdict = dict.fromkeys(["s40","s20","s-20","s-40","l40","l20","l-20","l-40","r40","r20","r-20","r-40"])
        
        [Tdict.update({key:dict.fromkeys(["flex","ext"],None)}) for key in Tdict.keys()]
        
        Tdict.update({"is_set": False})
        Q_MOCAP[file].update({"TBound":Tdict})
        
        "Kinematic analysis"
            
        
        fullc3dFileName = c3dfolder +  file;
        s = crd2dict (fullc3dFileName)
          
        
        HeadTraj = {'frontl' : s['MOCAP']['hlf'] , 'backl' : s['MOCAP']['hlb'] , 'frontr': s['MOCAP']['hrf'] , 'backr': s['MOCAP']['hrb']}; 
        ShouldTraj = {'SupL': s['MOCAP']['dlt'] , 'SupR': s['MOCAP']['drt'] , 'InfL' : s['MOCAP']['dlb'] , 'InfR': s['MOCAP']['drb']}; 
        #TopBarTraj = {'Cent': s['MOCAP']['d1c'] , 'Right': s['MOCAP']['d1r'] , 'Left': s['MOCAP']['d1l'] }
        
        globalFrame = np.eye(3); 
        
        Should_orient = get_orient(ShouldTraj , globalFrame , axis_form_square_arrangement_back);
        Head_orient  = get_orient(HeadTraj , globalFrame , axis_form_square_arrangement_head);
                
        #Head_orient = Head_orient         
        #Should_orient = np.full(Should_orient.shape,quaternion.quaternion(1,0,0,0))
            
        headYPR_abs = getYPR(Head_orient)
        bodyYPR_abs = getYPR(Should_orient)
        headYPR_rel = getYPR(Head_orient * Should_orient.conjugate())

        Q_MOCAP[file].update({"RPY": {"head_abs" : np.array(headYPR_abs) , "body_abs": np.array(bodyYPR_abs) , "head_rel" : np.array(headYPR_rel)}})
        
        Q_MOCAP[file]["EMG"] = s["EMG"]
        Q_MOCAP[file]["Mkr_samp"]["head"] = s['MOCAP']['hlf']
        Q_MOCAP[file]["Mkr_samp"]["back"] = s['MOCAP']['dlt']
        print("{} uploaded".format(file))
        
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
                             Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
                         
     Output:
         Orient: quaternion array [samples x ] with orientation of the body over time
    """
     
     Orient=MrkConf(Markers)
     
     Orient= quaternion.from_rotation_matrix(Orient) # array of quaternions [samples, Qdim]
     
     
     Orient=Orient * Orient[0].conjugate(); # Referering the head frames to the initial frame
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
                Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]] 
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
                Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
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
                Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
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

def set_boundaries(dict_data , seq):
    """
    This function ask for manually indicate the boundary points of the flexion
    and extension of the motion sequence. Such a data is stored in the dictionary.
    The clicks are from left to right starting from the beginnning and end of each neck motion
    

    Input:
    dict_data : dictionary that contains the MOCAP data
    seq : list discribing which type of motion correspond each pair of values
        "s" is for sagital motion , "c" for coronal motion and "r" for axial rotation

    """          
    nclicks = np.array(seq).size*4
    print("Make {} clicks".format(nclicks))
    boundaries=plt.ginput(nclicks,timeout=-1)
    boundaries=[bound[0] for bound in boundaries]
    boundaries=np.array(boundaries).reshape(-1,2,2)
    print (boundaries)
    for i , motion in enumerate(seq):
        regtry= dict_data["TBound"][motion]
        
        flag=[np.array(val == None).any() for val in regtry.values()].count(True)
        
        if flag:
            bound_upgr = list(boundaries[i,:,:])
            print("dict was empty")
        else:
            newext = np.append(regtry["ext"], boundaries[i,1,:])
            newflex = np.append(regtry["flex"], boundaries[i,0,:])         
            bound_upgr = [newflex , newext]   
            
        #bounds = {"ext":bound_upgr[0],"flex"bound_upgr[1]}
        regtry["ext"]=bound_upgr[1]
        regtry["flex"]=bound_upgr[0]
    dict_data["TBound"]["is_set"] = True      
    
def reset_data(MOCAP_EMG,File,save = True):
    Tdict = dict.fromkeys(["s","ll","lr","rl","rr"])
    [Tdict.update({key:dict.fromkeys(["flex","ext"],None)}) for key in Tdict.keys()]
    MOCAP_EMG[File].update({"TBound":Tdict})    
    MOCAP_EMG[File]['TBound']['is_set']=False
    if save:
        with open('preprocessed_dictionary.pkl', 'wb') as f:
            pickle.dump(MOCAP_EMG, f)

def show_DSET_map():
    pardir = os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))
    imdir = os.path.join(pardir,"EXT","Q_MOCAP_Structure.png")
    fig, ax = plt.subplots()
        
    image = plt.imread(imdir)    
    
    ax.imshow(image)
    ax.axis('off')

