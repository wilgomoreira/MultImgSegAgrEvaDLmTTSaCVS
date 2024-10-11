import tifffile
from PIL import Image
import tqdm
from torchvision.transforms import InterpolationMode
import torchvision.transforms as Tr
import util
import numpy as np
import os
import cv2
import multi_modal_transformation as T

AUGMENT = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(util.DATASET.RANDOM_HORIN_FLIP), 
                     T.AdjustSaturation(), T.AdjustBrightness()])
PREPROCESSING =  T.Compose([T.ToTensor()])

class Dataset:
    def __init__(self, **kwargs):
        self.root = kwargs['root']
        self.rgb_dir  = kwargs['rgb_dir']
        self.mask_dir = kwargs['mask_dir']
        self.set      = kwargs['set']
        self.num_classes = kwargs['num_classes']
        self.labels = util.DATASET.LABELS
        self.dataset_mode = util.DATASET.DATASET_MODE
        
        if 'dataset_mode' in kwargs:
            self.dataset_mode = kwargs['dataset_mode']

        assert self.dataset_mode in ['RAM','DISK']
        
        self.target_size = util.DATASET.TARGET_SIZE
        self.resize = Tr.Resize(self.target_size, interpolation=InterpolationMode.NEAREST)
        self.aug_bool = False
        
        if 'aug_bool' in kwargs:
            self.aug_bool = kwargs['aug_bool']
        
        self.aug = AUGMENT              
        self.preprocessing = PREPROCESSING
        self.rgb  = DatasetFiles(self.root, self.set, self.rgb_dir)
        self.mask = DatasetFiles(self.root, self.set, self.mask_dir)
        self.rgb_files  = self.rgb._get_files()
        self.mask_files = self.mask._get_files()
        self.dataset_len = len(self.rgb_files['files'])

        if self.dataset_mode == 'RAM':
            self. load_dataset_to_RAM()

    def __str__(self):
        return ' '.join([type(self).__name__ + '\n',
                        f'Loaded {self.dataset_len} Files\n',
                        f'Target size: {self.target_size}\n',
                        f'Loading mode: {self.dataset_mode}'
                        ])

    def load_dataset_to_RAM(self): 
        self.rgb_bag = []
        self.mask_bag =[]
        self.names_bag=[]

        for index in tqdm(range(self.dataset_len),"Loading to RAM: "):
            rgb, mask, id =  self.load_data(index)
            mask = _conv_img_to_mask_np(mask)
            rgb, mask = self.preprocessing(rgb,mask)
            mask[mask>0] = 1 
            self.rgb_bag.append(rgb)
            self.mask_bag.append(mask)
            self.names_bag.append(id)
           
    def get_data(self,indx):
        if self.dataset_mode == 'RAM':
            rgb = self.rgb_bag[indx]
            mask = self.mask_bag[indx]
            file_name= self.names_bag[indx]
        else:
            rgb, ndvi, gndvi, mask, file_name = self.load_data(indx)
            mask = _conv_img_to_mask_np(mask)
            rgb, mask = self.preprocessing(rgb, mask)
            mask[mask>0] = 1 
            mask = mask.squeeze(1) 

        return rgb, ndvi, gndvi, mask, file_name
    
    def load_data(self, index):
        NOT_ZERO = 1e-7
        
        mask_file, file_name = _build_file_path(self.mask_files, index)
        rgb_file, _ = _build_file_path(self.rgb_files, index)    
        mask = _load_file(mask_file)
        image = _load_file(rgb_file)

        nir = image[:, :, 4]
        r = image[:, :, 2]
        b = image[:, :, 0]
        g = image[:, :, 1]
        rgb = cv2.merge([r, g, b])
        
        ndvi = (nir.astype(float) - r.astype(float)) / (nir.astype(float) + r.astype(float) + NOT_ZERO)
        gndvi = (nir.astype(float) - g.astype(float)) / (nir.astype(float) + g.astype(float) + NOT_ZERO)

        return rgb, ndvi, gndvi, mask, file_name

    def set_aug_flag(self, flag):
        self.aug_bool = flag

    def __getitem__(self,index):
        rgb, ndvi, gndvi, mask, id =  self.get_data(index)
        
        if self.aug_bool:
            rgb, mask = self.aug(rgb, mask)

        return(rgb, ndvi, gndvi, mask, id)
    
    def __len__(self):
        return(self.dataset_len)
    
#---------------------------------------------------------------------------------------    
class DatasetFiles():
    def __init__(self, root, dataset, sensor, *kwargs):  
        assert os.path.isdir(root)
        assert dataset in ['altum','x7']
        assert sensor in util.DATASET.SENSORS

        self.root       = root
        self.set_dir    = dataset
        self.sensor_dir = sensor
        self.dir = os.path.join(self.root, self.set_dir, self.sensor_dir)
        
        if not os.path.isdir(self.dir):
            raise NameError("Path does not exist: " + self.dir)
 
    def _get_files(self):
        files = _get_files_in_dir(self.dir)
        
        if len(files) == 0: 
            raise ValueError("No Files found in: " + self.dir)

        file_name_clean = []
        file_type_clean = []
        
        for file in files:
            f,ftype = file.split('.')
            file_name_clean.append(f)
            file_type_clean.append(ftype)

        indices   = np.argsort(file_name_clean)
        files     = np.array(file_name_clean)[indices]
        file_type = np.array(file_type_clean)[indices]
        root_list = [self.dir]*len(file_type)
        
        return dict(root=root_list, files=files, type=file_type)
    
# Support Functions -----------------------------------------------------------------

def _conv_img_to_mask_np(img):
    if len(img.shape) == 3:  
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:  
        img_gray = img

    img_gray[img_gray < 128] = 0
    img_gray[img_gray >= 128] = 1

    return img_gray.astype(np.float32)
    
def _build_file_path(files, i): 
    file_name = np.array(files['files'])[i]
    file_type = np.array(files['type'])[i]
    root_file = np.array(files['root'])[i]
    file = os.path.join(root_file, file_name + '.' + file_type)
    
    return file, file_name

def _load_file(file): 
    file_type = file.split('.')[-1]
    
    if file_type == 'npy':
        im = np.load(file)
    elif file_type in ['JPG', 'PNG', 'jpg', 'png']:
        im = Image.open(file)
        im = np.array(im, dtype=np.uint8)
    else:
        im = tifffile.imread(file)
    
    return im

def _get_files_in_dir(folder_path):
    onlyfiles = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return onlyfiles 