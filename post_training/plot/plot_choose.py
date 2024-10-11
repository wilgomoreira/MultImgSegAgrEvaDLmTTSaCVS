import os
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
import torch
import pickle
import util

class PlotChoose:
    def __init__(self):
        self.chosen_images = util.PLOT.CHOOSE_IMAGES
        self.bands = util.PLOT.BANDS
        self.models = util.PLOT.MODELS
        self.start()

    def start(self):
        files_dic = self.transf_true_image_and_predict_roots()
        self.image_list = self.iterate_all_elements(files_dic)
        show_image = ShowImage(self.image_list)
        show_image.start_plot()

    def transf_true_image_and_predict_roots(self):
        true_image_root, predict_root = [], []
        fields = list(self.chosen_images.keys())
        number_fields = len(fields)

        path_dict = {}
        path_dicts_list = []

        for nField in range(number_fields):
            true_image_root = util.PLOT.CHOOSE_ROOTS['true_image'].replace('FIELD', fields[nField])
            path_dict.update({'true_image': true_image_root})
            for model in self.models:
                for band in self.bands:
                    predict_root = util.PLOT.CHOOSE_ROOTS['predict'].replace('MODEL', model).replace('INDEX', str(nField+1)).replace('BAND', band)
                    path_dict.update({f"{band}_{model}": predict_root})
            path_dicts_list.append(path_dict.copy())
        return self.join_fields_and_paths(path_dicts_list)

    def join_fields_and_paths(self, path_dicts_list):
        fields = list(self.chosen_images.keys())
        image_dic = {field: path for field, path in zip(fields, path_dicts_list)}
        return image_dic

    def iterate_all_elements(self, files_dic):
        pathFiles = []

        # when user don't choose images and the computer choose random images
        self.choosen_for_computer(files_dic)

        for fn, names in self.chosen_images.items():
            for name in names:
                for fr, root in files_dic.items():
                    if fr == fn:
                        pathFiles.append(self.join_root_withName(root, name))
        return pathFiles

    def choosen_for_computer(self, files_dic):
        for fn, names in self.chosen_images.items():
            for fr, root in files_dic.items():
                if not names:
                    if fn == fr:
                        random_name = self.images_random(root)
                        self.chosen_images[fn] = [random_name]

    def images_random(self, root):
        path = root['true_image']
        files = [name for name in os.listdir(path)]
        return random.choice(files)

    def join_root_withName(self, root, file_name):
        pathFile = {}

        for key, value in root.items():
            if key == 'true_image':
                pathFile[key] = value + file_name
            else:
                pathFile[key] = (value + file_name).replace(util.PLOT.CHOOSE_FILE_EXT['tif'], util.PLOT.CHOOSE_FILE_EXT['pickle']).replace(util.PLOT.CHOOSE_FILE_EXT['npy'], util.PLOT.CHOOSE_FILE_EXT['pickle'])
        return pathFile


#########################################################################################
class ShowImage:
    def __init__(self, image_list):
        self.image_list = image_list
  
    def start_plot(self):
        n_rows = len(self.image_list)
        n_cols = len(self.image_list[0])
  
        self.prepare_and_plot(n_rows, n_cols)
    
    def prepare_and_plot(self, n_rows, n_cols):
        models, titles = [], []
  
        for i in range(n_rows):
            for key, file in self.image_list[i].items():
                image = self.get_image(file)
                models.append(image)
                titles.append(key)

        self.plot(models, titles, n_rows, n_cols)
  
    def get_image(self, file):  
        if util.PLOT.CHOOSE_FILE_EXT['tif'] in file:
            return self.read_tif(file)
        if util.PLOT.CHOOSE_FILE_EXT['npy'] in file:
            return self.read_npy(file)
        if util.PLOT.CHOOSE_FILE_EXT['pickle'] in file:
            return self.read_pickle(file)

    def read_tif(self, file):
        image = cv2.imread(file)
        return image

    def read_npy(self, file):
        image = np.load(file)
        return image

    def read_pickle(self, file):
        opened_pkl = open(file, 'rb')
        dic_image = pickle.load(opened_pkl)
        pickle_image = dic_image['pred']
        inv_image = torch.from_numpy(pickle_image)
        image = torch.permute(inv_image,(1,2,0))
        return image
  
    def plot(self, models, titles, n_rows, n_cols):
        fig = plt.figure()
        fig.set_size_inches(util.PLOT.CHOOSE_SUBPLOT_SIZE)
        index = 1
  
        for model, title in zip(models, titles):
            plt.subplot(n_rows,n_cols,index)
            plt.title(title)
            plt.imshow(model)
            plt.xticks([]),plt.yticks([])
            plt.savefig(util.PLOT.CHOOSE_SAVE_IMAGE)
            index +=1