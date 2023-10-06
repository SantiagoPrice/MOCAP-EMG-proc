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
         .Markers: struct with the fields: frontl, backl, leftr, backr 
         Each field is a [sample x 3] matrix with each marker temporal evolution

        . Output:
            RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
     """    
     u = (Markers['frontl'] - Markers['backl'])+(Markers['frontr'] - Markers['backr'])
     #u=(Markers['frontl'] - Markers['frontr'])+(Markers['backl'] - Markers['backr'])
     unorm = np.linalg.norm(u,axis=1).reshape(-1,1)
     u = u / unorm;
     
     
     
     v = (Markers['frontl'] - Markers['frontr'])+(Markers['backl'] - Markers['backr'])
     #v= (Markers['frontl'] - Markers['backl'])+(Markers['frontr'] - Markers['backr'])
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
         .Markers: struct with the fields: frontl, backl, backr 
         Each field is a [sample x 3] matrix with each marker temporal evolution

        . Output:
            RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
     """    
     u = (Markers['frontl'] - Markers['backl'])
     unorm = np.linalg.norm(u,axis=1).reshape(-1,1)
     u = u / unorm;
     
     
     
     v = (Markers['backl'] - Markers['backr'])
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
           .Markers: struct with the fields: frontr, backl, backr 
           Each field is a [sample x 3] matrix with each marker temporal evolution

          . Output:
              RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                  Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
       """    
       u = (Markers['frontr'] - Markers['backr'])
       unorm = np.linalg.norm(u,axis=1).reshape(-1,1)
       u = u / unorm;
       
       
       
       v = (Markers['backl'] - Markers['backr'])
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
          .Markers: struct with the fields: frontl, backl, backr 
          Each field is a [sample x 3] matrix with each marker temporal evolution

         . Output:
             RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                 Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
      """    
      u = (Markers['frontl'] - Markers['backl'])
      unorm = np.linalg.norm(u,axis=1).reshape(-1,1)
      u = u / unorm;
      
      
      
      v = (Markers['frontl'] - Markers['frontr'])
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
     w = w / wnorm;
     
     
     
     v = (Markers['SupL'] - Markers['SupR'])+(Markers['InfL'] - Markers['InfR'])
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
          .Markers: struct with the fields: SupL, Supr, InfR, InfR 
          Each field is a [sample x 3] matrix with each marker temporal evolution

         . Output:
             RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                 Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
      """    
      w = (Markers['SupR'] - Markers['InfR'])
      wnorm = np.linalg.norm(w,axis=1).reshape(-1,1)
      w = -w / wnorm;
      
      
      
      v = (Markers['InfL'] - Markers['InfR'])
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
          .Markers: struct with the fields: SupL, Supr, InfR, InfR 
          Each field is a [sample x 3] matrix with each marker temporal evolution

         . Output:
             RefLocal: framework uvw of the body over time.[frames x 3 x 3] Matrix
                 Example: RefLocal[i,:,:]=[[ux,uy,uz],[vx,vy,vz],[wx,wy,wz]]
      """    
      w = (Markers['SupL'] - Markers['InfL'])
      wnorm = np.linalg.norm(w,axis=1).reshape(-1,1)
      w = -w / wnorm;
      
      
      
      v = (Markers['SupL'] - Markers['SupR'])
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
  
def YPR_from_mrks():
    pass