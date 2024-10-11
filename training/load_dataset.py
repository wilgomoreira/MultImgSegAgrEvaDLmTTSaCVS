from dataclasses import dataclass
from dataset import Dataset
import util

@dataclass
class LoadDataset:
    loader_val: Dataset
    loader_esac: Dataset
    loader_qbaixo: Dataset
    
    @classmethod
    def from_files(clc, dataset_name):
        num_classes = util.NUM_CLASSES
        root_val = util.DATASET_ROOT.VALDOEIRO
        root_esac = util.DATASET_ROOT.ESAC
        root_qbaixo = util.DATASET_ROOT.QBAIXO
        
        set_ = util.DATASET.SET
        rgb_dir = util.DATASET.RGB_DIR
        mask_dir = util.DATASET.MASK_DIR
        
        assert dataset_name in [util.T1, util.T2, util.T3],'Dataset not recognized'
        
        loader_val = Dataset(root=root_val, set=set_, rgb_dir=rgb_dir, 
                             mask_dir=mask_dir, num_classes=num_classes) 
        loader_esac = Dataset(root=root_esac, set=set_, rgb_dir=rgb_dir, 
                              mask_dir=mask_dir, num_classes=num_classes)
        loader_qbaixo = Dataset(root=root_qbaixo, set=set_, rgb_dir=rgb_dir, 
                                mask_dir=mask_dir, num_classes=num_classes)
        
        return clc(loader_val=loader_val, loader_esac=loader_esac, loader_qbaixo=loader_qbaixo)
        
        

    


