import os
import numpy as np
from utils import Spring
#from smpl.serialization import load_model
from DataReader.read import DataReader, quads2tris
from DataReader.IO import writeOBJ
from DataReader.smpl.smpl_np import SMPLModel
    

if __name__=='__main__':
    
	# SMPL model
    smpl_path = os.path.join('/home/cxh/Documents/sources/ClothDeform/assets','SMPL')
    smpl_model = {
	        'f': SMPLModel(os.path.join(smpl_path, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')),
			'm': SMPLModel(os.path.join(smpl_path, 'basicModel_m_lbs_10_207_0_v1.0.0.pkl'))}
    pose = np.random.normal(0.0, 0.2, 72)*0.5
    shape = np.random.normal(0.0, 0.1, 10)
    vertices, joints = smpl_model['f'].set_params(pose, shape,trans=np.array([0,0,0]))
    faces = smpl_model['f'].faces 
    weights = smpl_model['f'].weights
    print(f' weights : {weights.shape}')
    
    print('Done')

