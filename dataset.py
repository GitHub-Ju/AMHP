import os
import torch
import functools
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import clip

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

ImageFile.LOAD_TRUNCATED_IMAGES = True
def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def image_loader(image_name):
    if has_file_allowed_extension(image_name, IMG_EXTENSIONS):
        I = Image.open(image_name)
    return I.convert('RGB')

def get_default_img_loader():
    return functools.partial(image_loader)

class ImageDataset(Dataset):
    def __init__(self, csv_file,
                 img_dir,
                 preprocess,
                 num_patch,
                 test,
                 get_loader=get_default_img_loader):
        self.data = pd.read_csv(csv_file)
        print('%d csv data successfully loaded!' % self.__len__())
        self.img_dir = img_dir
        self.loader = get_loader()
        self.preprocess = preprocess
        self.num_patch = num_patch
        self.test = test

    def __getitem__(self, index):

        row = self.data.iloc[index]
        image_name = os.path.join('/media/boot/4T/dataset/PARA/imgs', os.path.join(row["sessionId"], row["imageName"]))

        I = self.loader(image_name)
        I = self.preprocess(I)
        I = I.unsqueeze(0)
        n_channels = 3
        kernel_h = 224
        kernel_w = 224
        if (I.size(2) >= 1024) | (I.size(3) >= 1024):
            step = 48
        else:
            step = 8
        patches = I.unfold(2, kernel_h, step).unfold(3, kernel_w, step).permute(2, 3, 0, 1, 4, 5).reshape(-1,
                                                                                                          n_channels,
                                                                                                          kernel_h,
                                                                                                          kernel_w)

        assert patches.size(0) >= self.num_patch
        if self.test:
            sel_step = patches.size(0) // self.num_patch
            sel = torch.zeros(self.num_patch)
            for i in range(self.num_patch):
                sel[i] = sel_step * i
            sel = sel.long()
        else:
            sel = torch.randint(low=0, high=patches.size(0), size=(self.num_patch, ))
        patches = patches[sel, ...]


        aes_mean = row["aestheticScore_mean"]

        aes1 = row["aestheticScore_1.0"]
        aes2 = row["aestheticScore_1.5"] + row["aestheticScore_2.0"]
        aes3 = row["aestheticScore_2.5"] + row["aestheticScore_3.0"]
        aes4 = row["aestheticScore_3.5"] + row["aestheticScore_4.0"]
        aes5 = row["aestheticScore_5.0"]
        aes_distri = [aes1, aes2, aes3, aes4, aes5]
        p = np.array(aes_distri)
        p = p / np.sum(p)

        # composition
        comp_mean = row["compositionScore_mean"] 
        comp_level = ''
        if comp_mean <= 1:
            comp_level = 'bad'
        if 1 < comp_mean <= 2:
            comp_level = 'poor'
        if 2 < comp_mean <= 3:
            comp_level = 'fair'
        if 3 < comp_mean <= 4:
            comp_level = 'good'
        if comp_mean > 4:
            comp_level = 'perfect'

        # color
        color_mean = row["colorScore_mean"]
        color_level = ''
        if color_mean <= 1:
            color_level = 'bad'
        if 1 < color_mean <= 2:
            color_level = 'poor'
        if 2 < color_mean <= 3:
            color_level = 'fair'
        if 3 < color_mean <= 4:
            color_level = 'good'
        if color_mean > 4:
            color_level = 'perfect'

        # content
        cont_mean = row["contentScore_mean"]
        cont_level = ''
        if cont_mean <= 1:
            cont_level = 'bad'
        if 1 < cont_mean <= 2:
            cont_level = 'poor'
        if 2 < cont_mean <= 3:
            cont_level = 'fair'
        if 3 < cont_mean <= 4:
            cont_level = 'good'
        if cont_mean > 4:
            cont_level = 'perfect'

        # light
        light_mean = row["lightScore_mean"]
        light_level = ''
        if light_mean <= 1:
            light_level = 'bad'
        if 1 < light_mean <= 2:
            light_level = 'poor'
        if 2 < light_mean <= 3:
            light_level = 'fair'
        if 3 < light_mean <= 4:
            light_level = 'good'
        if light_mean > 4:
            light_level = 'perfect'

        # DOF
        dof_mean = row["dofScore_mean"]
        dof_level = ''
        if dof_mean <= 1:
            dof_level = 'bad'
        if 1 < dof_mean <= 2:
            dof_level = 'poor'
        if 2 < dof_mean <= 3:
            dof_level = 'fair'
        if 3 < dof_mean <= 4:
            dof_level = 'good'
        if dof_mean > 4:
            dof_level = 'perfect'


        # OB
        iob_mean = row["isObjectEmphasis_mean"]
        iob_level = ''
        if iob_mean <= 0.2:
            iob_level = 'bad'
        if 0.2 < iob_mean <= 0.4:
            iob_level = 'poor'
        if 0.4 < iob_mean <= 0.6:
            iob_level = 'fair'
        if 0.6 < iob_mean <= 0.8:
            iob_level = 'good'
        if iob_mean > 0.8:
            iob_level = 'perfect'

        att_texts = f"A photo of {cont_level} content with {comp_level} composition and {color_level} color, which is of {light_level} light and {dof_level} depth of field and {iob_level} object emphasis."


        sample = {
            'I': patches,
            'aes_mean': float(aes_mean),
            'aes_distri': p.astype('float32'),
            'att_texts': att_texts,
        }

        return sample

    def __len__(self):
        return len(self.data.index)
