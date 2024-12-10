# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:04:48 2023

@author: UTAIL
"""
import numpy as np

# def make_base_generic(Markers, Fo):
#     """
#     This function makes an ortonormal base from the markers using a fiven criteria
#     Input:
#         .Markers: struct with the fields: front, back, left, right 
#         Each field is a [sample x 3] matrix with each marker temporal evolution

#        . Output:
#            RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
#                Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]] 
#     """

def ortonorm_vspace(v1,v2):
    
    v1norm=np.linalg.norm(v1,axis=1).reshape(-1,1)
    v1 = v1 / v1norm;
    
    proj = np.sum(v2*v1,axis=1).reshape(-1,1);         #projection of v2 in v1 or viceversa
    v2 = (v2- proj * v1);                 # removing V2 component in the v1 direction
    v2norm = np.linalg.norm(v2,axis=1).reshape(-1,1)
    v2 = v2 / v2norm
    
    v3 = np.cross(v1,v2);
    v3norm = np.linalg.norm(v3,axis=1).reshape(-1,1)
    v3 = v3 / v3norm; 
    
    
    v1=v1.reshape(-1,1,3) 
    v2=v2.reshape(-1,1,3) 
    v3=v3.reshape(-1,1,3)
    
    return v1, v2, v3
   
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
       
     v = Markers['left'] - Markers['right'];


     u,v,w= ortonorm_vspace(u,v)
     RefLocal= np.hstack((u,v,w));               # Transformation matrix array [samples , xyz glob(3) , xyz loc(3), ]
     RefLocal= np.swapaxes(RefLocal,1,2)
     return RefLocal

def axis_form_square_arrangement_head(Markers):
     """
     This function makes an ortonormal base from four markers in the square arrangement like in the RoboSoft experiment
     Input:
         .Markers: struct with the fields: hlf, hlb, leftr, hrb 
         Each field is a [sample x 3] matrix with each marker temporal evolution

        . Output:
            RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
     """    
     u = (Markers['hlf'] - Markers['hlb'])+(Markers['hrf'] - Markers['hrb'])
   
     
     v = (Markers['hlf'] - Markers['hrf'])+(Markers['hlb'] - Markers['hrb'])
     u,v,w= ortonorm_vspace(u,v) 
     RefLocal= np.hstack((u,v,w));               # Transformation matrix array [samples , xyz glob(3) , xyz loc(3), ]      
     
     return RefLocal
 

def axis_form_L_arrangement_head(Markers):
     """
     This function makes an ortonormal base from three markers arrangement
     Input:
         .Markers: struct with the fields: hlf, hlb, hrb 
         Each field is a [sample x 3] matrix with each marker temporal evolution

        . Output:
            RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
     """    
     u = (Markers['hlf'] - Markers['hlb'])
       
     v = (Markers['hlb'] - Markers['hrb'])
     
     u,v,w= ortonorm_vspace(u,v) 
     RefLocal= np.hstack((u,v,w));               # Transformation matrix array [samples , xyz glob(3) , xyz loc(3), ]
     RefLocal= np.swapaxes(RefLocal,1,2)
     return RefLocal    
 
def axis_form_L_arrangement_head_inv(Markers):
       """
       This function makes an ortonormal base from three markers arrangement
       Input:
           .Markers: struct with the fields: hrf, hlb, hrb 
           Each field is a [sample x 3] matrix with each marker temporal evolution

          . Output:
              RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                  Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
       """    
       u = (Markers['hrf'] - Markers['hrb'])
             
       v = (Markers['hlb'] - Markers['hrb'])

       u,v,w= ortonorm_vspace(u,v)     
       RefLocal= np.hstack((u,v,w));               # Transformation matrix array [samples , xyz glob(3) , xyz loc(3), ]
       RefLocal= np.swapaxes(RefLocal,1,2)
       return RefLocal    
 
def axis_form_L_arrangement_head_invv(Markers):
      """
      This function makes an ortonormal base from three markers arrangement
      Input:
          .Markers: struct with the fields: hlf, hlb, hrb 
          Each field is a [sample x 3] matrix with each marker temporal evolution

         . Output:
             RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                 Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
      """    
      u = (Markers['hlf'] - Markers['hlb'])
   
      v = (Markers['hlf'] - Markers['hrf'])


      u,v,w= ortonorm_vspace(u,v) 
      RefLocal= np.hstack((u,v,w));               # Transformation matrix array [samples , xyz glob(3) , xyz loc(3), ]
      RefLocal= np.swapaxes(RefLocal,1,2)
      return RefLocal     
 
def axis_form_square_arrangement_back (Markers):
     """
     This function makes an ortonormal base from four markers in the square arrangement like in the RoboSoft experiment
     Input:
         .Markers: struct with the fields: dlt, drt, drb, drb 
         Each field is a [sample x 3] matrix with each marker temporal evolution

        . Output:
            RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
     """    
     w = (Markers['dlt'] - Markers['dlb'])+(Markers['drt'] - Markers['drb'])

     
     
     v = (Markers['dlt'] - Markers['drt'])+(Markers['dlb'] - Markers['drb'])

     

     
     w,v,u= ortonorm_vspace(w,v) 
     RefLocal= np.hstack((u,v,w));               # Transformation matrix array [samples , xyz glob(3) , xyz loc(3), ]
     #RefLocal= np.swapaxes(RefLocal,1,2)
     return RefLocal
 
    
def axis_form_L_arrangement_back_inv(Markers):
      """
      This function makes an ortonormal base from from three markers arrangement
      Input:
          .Markers: struct with the fields: dlt, drt, drb, drb 
          Each field is a [sample x 3] matrix with each marker temporal evolution

         . Output:
             RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                 Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
      """    
      w = (Markers['drt'] - Markers['drb'])

      
      
      v = (Markers['dlb'] - Markers['drb'])

      
      
      w,v,u= ortonorm_vspace(w,v)     
      RefLocal= np.hstack((-u,v,-w));               # Transformation matrix array [samples , xyz glob(3) , xyz loc(3), ]
      RefLocal= np.swapaxes(RefLocal,1,2)
      return RefLocal
    
def axis_form_L_arrangement_back(Markers):
      """
      This function makes an ortonormal base from from three markers arrangement
      Input:
          .Markers: struct with the fields: dlt, drt, drb, drb 
          Each field is a [sample x 3] matrix with each marker temporal evolution

         . Output:
             RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                 Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
      """    
      w = (Markers['dlt'] - Markers['dlb'])
     
      v = (Markers['dlt'] - Markers['drt'])

    
      w,v,u= ortonorm_vspace(w,v)     
      RefLocal= np.hstack((-u,v,-w));               # Transformation matrix array [samples , xyz glob(3) , xyz loc(3), ]
      return RefLocal 

def axis_form_L_arrangement_back2(Markers):
      """
      This function makes an ortonormal base from from three markers arrangement
      Input:
          .Markers: struct with the fields: dlt, drt, drb, drb 
          Each field is a [sample x 3] matrix with each marker temporal evolution

         . Output:
             RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                 Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
      """    
      w = (Markers['dlt'] - Markers['dlb'])
   
      v = (Markers['dlb'] - Markers['drb'])

      
      w,v,u= ortonorm_vspace(w,v)     
      RefLocal= np.hstack((-u,v,-w));               # Transformation matrix array [samples , xyz glob(3) , xyz loc(3), ]
      return RefLocal 

def axis_form_square_arrangement_head_EMBC24(Markers):
     """
     This function remaps the markers from EMBC experiment to the Robosoft configuration 
     Input:
         .Markers: struct with the fields: hfl, hbl, hfr, hbr 
         Each field is a [sample x 3] matrix with each marker temporal evolution

        . Output:
            RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
     """
     
     u = (Markers['hfl'] - Markers['hbl'])+(Markers['hfr'] - Markers['hbr'])
   
     v = (Markers['hfl'] - Markers['hfr'])+(Markers['hbl'] - Markers['hbr'])
     
     u,v,w= ortonorm_vspace(u,v) 
     RefLocal= np.hstack((u,v,w));               # Transformation matrix array [samples , xyz glob(3) , xyz loc(3), ]     
     return RefLocal
 

def axis_form_square_arrangement_back_EMBC24(Markers):
     """
     This function makes an ortonormal base from four markers in the square arrangement like in the RoboSoft experiment
     Input:
         .Markers: struct with the fields:tlb tlt trb trt
             dlt, drt, drb, dlb 
         Each field is a [sample x 3] matrix with each marker temporal evolution

        . Output:
            RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
     """
     
     w = (Markers['tlt'] - Markers['tlb'])+(Markers['trt'] - Markers['trb'])
    
     v = (Markers['tlt'] - Markers['trt'])+(Markers['tlb'] - Markers['trb'])
   
     w,v,u= ortonorm_vspace(w,v) 
     RefLocal= np.hstack((u,v,w));               # Transformation matrix array [samples , xyz glob(3) , xyz loc(3), ]
     #RefLocal= np.swapaxes(RefLocal,1,2)
     return RefLocal

      
def g_ant_rf(markers): 
    pass

def g_ant_lf(markers): 
    pass   

def g_ant_hip(markers): 
    names_map={'hlf':"LASI",'hrf':"RASI",'hlb':"LPSI",'hrb':"RPSI"} 
    Markers=dict()
    for mk in names_map:
       old_name=names_map[mk]
       Markers.update({mk:markers[old_name]})
  
    return axis_form_square_arrangement_head(Markers)

def g_ant_shoul(markers): 
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
    
    v = markers['LSHO'] - markers['RSHO'];
    vnorm = np.linalg.norm(v,axis=1).reshape(-1,1)
    v = v / vnorm;
    
    w = np.zeros(v.shape);
    w[:,2]=1
   
    
    u = np.cross(v,w)
    
    u=u.reshape(-1,1,3) 
    v=v.reshape(-1,1,3) 
    w=w.reshape(-1,1,3)      
    RefLocal= np.hstack((u,v,w));               # Transformation matrix array [samples , xyz glob(3) , xyz loc(3), ]
    RefLocal= np.swapaxes(RefLocal,1,2)
    return RefLocal

def g_ant_head(markers):
     names_map={'hlf':"LFHD",'hrf':"RFHD",'hlb':"LBHD",'hrb':"RBHD"} 
     Markers=dict()
     for mk in names_map:
        old_name=names_map[mk]
        Markers.update({mk:markers[old_name]})
   
     return axis_form_square_arrangement_head(Markers)

def DHS_head(Markers):  
     print("head")
     u = (Markers["hfl"] - Markers["hbl"])+(Markers["hfr"] - Markers["hbr"])
     
     v = (Markers["hfl"] - Markers["hfr"])+(Markers["hbl"] - Markers["hbr"])
     
     u,v,w= ortonorm_vspace(u,v) 
     RefLocal= np.hstack((u,v,w));               # Transformation matrix array [samples , xyz glob(3) , xyz loc(3), ]      
     
     return RefLocal


def DHS_trunk(Markers):
    print("trunk")
    
    for m in Markers.items():
         if np.isnan(m[1]).any():
             print(f"{m[0]} has nan vaues")
             
    w = (Markers["t3"] - Markers["t1"])
 
    v = (Markers["t1"] - Markers["t2"])
 
    w,v,u= ortonorm_vspace(w,v)     
    RefLocal= np.hstack((-u,v,-w));               # Transformation matrix array [samples , xyz glob(3) , xyz loc(3), ]
    RefLocal= np.swapaxes(RefLocal,1,2)
    return RefLocal
    
    
def DHS_act_tip(markers):
     print("actuator")
     
             
     u = markers['ac'] - markers['al']
   
     v =markers['ac'] - markers['ar']
     
     u,v,w= ortonorm_vspace(u,v) 
     RefLocal= np.hstack((u,v,w));               # Transformation matrix array [samples , xyz glob(3) , xyz loc(3), ]     
     return RefLocal
     
    

def YPR_from_mrks():
    pass