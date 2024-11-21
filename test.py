import os
import numpy as np
from utils import Spring
from typing import List,Dict
import yaml
#from smpl.serialization import load_model
from DataReader.read import DataReader, quads2tris
from DataReader.IO import writeOBJ
from DataReader.smpl.smpl_np import SMPLModel
import h5py
import logging
import pickle


def cloth3d(indices:int, config:Dict)->List[Dict]:
    r'''
    Extract data and store the data into pickle file
    '''
    reader = DataReader(config)
    list_data = []
    # Create dict with keys
    data_keys = ['trans', 'shape', 'pose', 'gender', 'garment', 'fabric', 'garment_vertices', 'garment_faces', 'human_vertices', 'human_faces']
    #data_dict = dict.fromkeys(data_keys,[]*10)
    data_dict = {key:[] for key in data_keys}
    total = 0
    for i in range(indices):
        sample = '{0:0>5}'.format(i)
        logging.debug(f' Reading data : {sample}...')
        info = reader.read_info('{0:0>5}'.format(sample))

        # Human metadata
        # SMPL root joint location (3, #frames) 
        # Root joint is first aligned at (0,0,0) and later moved to its corresponding location. 
        # SMPL does not align root joint at origin by default.
        if info['trans'].ndim==2:
            trans = info['trans'][:,:]
            num_frame = info['trans'].shape[-1] # sequence length
        else:
            trans = info['trans']
        
        shape = info['shape']       # SMPL shape parameters (10,)
        if info['poses'].ndim==2:
            poses = info['poses'][:,:]  # SMPL pose parameters (72, #frames)
        else:
            poses = info['poses']
        
        gender = info['gender']     # subject's gender: 0: female, 1: male
        # Garment metadata
        garments = list(info['outfit'].keys())
        #logging.debug(f'outfit: {garments} ')
        outfit = info['outfit']
        #tightness = info['tightness']
        for j, garment in enumerate(garments):
            fabric = info['outfit'][garment]['fabric']
            g_f = reader.read_garment_topology(sample=sample, garment=garment)
            
            g_f = quads2tris(g_f)
            for fm in range(num_frame):
                g_v = reader.read_garment_vertices(sample=sample, garment=garment, frame=fm)
                data_dict['garment_vertices'].append(g_v)
                h_v, h_f = reader.read_human(sample=sample, frame=fm)
                data_dict['human_vertices'].append(h_v)
            data_dict['trans'].append(trans)
            data_dict['shape'].append(shape)
            data_dict['pose'].append(poses)
            data_dict['gender'].append(gender)
            data_dict['garment'].append(garment)
            data_dict['fabric'].append(fabric)
            
            data_dict['garment_faces'].append(g_f)
            
            data_dict['human_faces'].append(h_f)
            list_data.append(data_dict)
        #logging.debug(' pose shape : {0}'.format(data_dict['pose'].shape))
        logging.debug(f' Done.')
    return list_data 

class Cloth3D():
    def __init__(self, config:Dict):
        self.reader = DataReader(config=config)
        # create dict with keys
        data_keys = ['trans', 'shape', 'pose', 'gender', 'garment', 
                    'fabric', 'garment_vertices', 'garment_faces', 'human_vertices', 'human_faces']
        self.data_dict = {key:[] for key in data_keys}
        #print( 'data dict : {0}'.format(data_dict))
        

    def get_item(self, index:int):
        self.info = self.reader.read_info('{0:0>5}'.format(index))
        self.sample = '{0:0>5}'.format(index)
        #logging.debug('trans dim : {0}'.format(self.info['trans'].shape))
        self.trans = np.array(self.info['trans'],dtype=np.float32)
        #logging.debug(' trans : {0}'.format(self.trans))
        self.shape = self.info['shape']
        self.poses = self.info['poses']
        self.gender = self.info['gender']
        self.num_frame=self.trans.shape[-1] # 300 frames
        logging.debug(' num frame : {0}'.format(self.num_frame))
        self.garments = list(self.info['outfit'].keys())
        self.outfit = self.info['outfit']

        for j, garment in enumerate(self.garments):
            logging.debug(' garment : {0}'.format(garment))
            self.fabric = self.info['outfit'][garment]['fabric']
            #logging.debug(' fabric : {0}'.format(self.fabric))
            g_f = self.reader.read_garment_topology(sample=self.sample, garment=garment)
            g_f = quads2tris(g_f)
            garment_vertices = []
            for fm in range(self.num_frame):
                g_v = self.reader.read_garment_vertices(sample=self.sample,garment=garment,frame=fm)
                #g_v = np.array(g_v,dtype=np.float32)
                #logging.debug('gv : {0}'.format(g_v))
                garment_vertices.append(g_v)
                #print(' garmetn vertices : {0}')
                #self.data_dict['garment_vertices'].append(g_v)
                #logging.debug(' g_v shape : {0}'.format(garment_vertices))
                h_v, h_f = self.reader.read_human(sample=self.sample,frame=fm)
                #self.data_dict['human_vertices'].append(h_v)

            garment_vertices = np.array(garment_vertices,dtype=np.float32)
            logging.debug(' garment vertices shape : {0}'.format(garment_vertices.shape))



if __name__=='__main__':
    logging.basicConfig(level=logging.DEBUG)
    with open(file='config.yaml', mode='r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    indices = np.loadtxt(config['val_indices'], dtype=np.int32)
    logging.debug(f' indices : {indices}')
    cloth_data = Cloth3D(config)
    for i in indices:
        cloth_data.get_item(i)
    #print(' data item: {0}'.format(cloth_data))
    
            
    
    print('Done')
