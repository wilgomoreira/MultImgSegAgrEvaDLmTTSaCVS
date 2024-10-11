from dataclasses import dataclass
import numpy as np
from pathlib import Path
import util
import pickle

@dataclass
class Image:
    name: str
    model: str
    database: str
    spectrum: str
    value: np.ndarray
     
    @classmethod
    def from_file(clc, name, model, database, spectrum=util.DEFAULT_SPEC):
        file_paths = _build_path(model=model.lower(), database=database.lower(), spectrum=spectrum.upper())  
        values = _get_values(file_paths=file_paths, name=name)
    
        return clc(name=name, model=model, database=database, spectrum=spectrum, value=values)
         
def _build_path(model, database, spectrum):
    parent = Path(util.DIR.PARENT)
    child = Path(f"{model.lower()}_{database.lower()}_{spectrum.upper()}")
    grandson_train = Path(util.DIR.GRANDSON_TRAIN)
    grandson_test = Path(util.DIR.GRANDSON_TEST)
    
    file_path_train = parent / child / grandson_train
    file_path_test = parent / child / grandson_test
    
    assert file_path_train.exists(), "Path does not exist"   
    assert file_path_test.exists(), "Path does not exist"   
    return file_path_train, file_path_test

def _get_values(file_paths, name):
    value_train, value_test = [], []
    file_path_train, file_path_test = file_paths
    
    for file in file_path_train.glob(util.DIR.EXTENSION):
        samples = _read_pickle(file, name) 
        if name == util.PRED:
            samples = samples.cpu()
        value_train.append(samples)

    for file in file_path_test.glob(util.DIR.EXTENSION):
        samples = _read_pickle(file, name) 
        if name == util.PRED:
            samples = samples.cpu()
        value_test.append(samples)

    value_train = np.array(value_train)  
    value_test = np.array(value_test)
    
    return value_train, value_test
                       
def _read_pickle(file, name):   
    with open(file, 'rb') as pkl_file:
        param_dic = pickle.load(pkl_file)     
        return param_dic[name.lower()]   
