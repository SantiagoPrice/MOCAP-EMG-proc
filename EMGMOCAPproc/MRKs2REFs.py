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
         .Markers: struct with the fields: hlf, hlb, leftr, hrb 
         Each field is a [sample x 3] matrix with each marker temporal evolution

        . Output:
            RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
     """    
     u = (Markers['hlf'] - Markers['hlb'])+(Markers['hrf'] - Markers['hrb'])
     #u=(Markers['hlf'] - Markers['hrf'])+(Markers['hlb'] - Markers['hrb'])
     unorm = np.linalg.norm(u,axis=1).reshape(-1,1)
     u = u / unorm;
     
     
     
     v = (Markers['hlf'] - Markers['hrf'])+(Markers['hlb'] - Markers['hrb'])
     #v= (Markers['hlf'] - Markers['hlb'])+(Markers['hrf'] - Markers['hrb'])
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
     #RefLocal= np.swapaxes(RefLocal,1,2)
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
     unorm = np.linalg.norm(u,axis=1).reshape(-1,1)
     u = u / unorm;
     
     
     
     v = (Markers['hlb'] - Markers['hrb'])
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
       unorm = np.linalg.norm(u,axis=1).reshape(-1,1)
       u = u / unorm;
       
       
       
       v = (Markers['hlb'] - Markers['hrb'])
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
      unorm = np.linalg.norm(u,axis=1).reshape(-1,1)
      u = u / unorm;
      
      
      
      v = (Markers['hlf'] - Markers['hrf'])
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
         .Markers: struct with the fields: dlt, drt, drb, drb 
         Each field is a [sample x 3] matrix with each marker temporal evolution

        . Output:
            RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
     """    
     w = (Markers['dlt'] - Markers['dlb'])+(Markers['drt'] - Markers['drb'])
     wnorm = np.linalg.norm(w,axis=1).reshape(-1,1)
     w = w / wnorm;
     
     
     
     v = (Markers['dlt'] - Markers['drt'])+(Markers['dlb'] - Markers['drb'])
     proj = np.sum(v*w,axis=1).reshape(-1,1);         #projection of v in w or viceversa
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
      wnorm = np.linalg.norm(w,axis=1).reshape(-1,1)
      w = -w / wnorm;
      
      
      
      v = (Markers['dlb'] - Markers['drb'])
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
      wnorm = np.linalg.norm(w,axis=1).reshape(-1,1)
      w = -w / wnorm;
      
      
      
      v = (Markers['dlt'] - Markers['drt'])
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
      #RefLocal= np.swapaxes(RefLocal,1,2)
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
     currLab=["hfl", "hbl", "hbr", "hbc" ]
     newLab=["hlf", "hlb", "hrf", "hrb" ]
     relab_mocap=dict()
     for nl , ol in zip(newLab,currLab):     
         relab_mocap.update({nl:Markers[ol]})
     Markers=relab_mocap
     return axis_form_square_arrangement_head(Markers)
 

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
     currLab=["tlt", "trt", "trb", "tlb" ]
     newLab=["dlt", "drt", "drb", "dlb" ]
     relab_mocap=dict()
     for nl , ol in zip(newLab,currLab):     
         relab_mocap.update({nl:Markers[ol]})
     Markers=relab_mocap
     return axis_form_square_arrangement_back(Markers)
      
         

def YPR_from_mrks():
    pass