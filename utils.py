import torch
from torch.utils.data import DataLoader
from dataset import ImageDataset

from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, Resize
from torchvision import transforms

from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def set_dataset(csv_file, bs, data_set, num_workers, preprocess, num_patch, test):

    data = ImageDataset(
        csv_file=csv_file,
        img_dir=data_set,
        num_patch=num_patch,
        test=test,
        preprocess=preprocess)

    if test:
        shuffle = False
    else:
        shuffle = True

    loader = DataLoader(data, batch_size=bs, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return loader



class AdaptiveResize(object):
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, image_size=None):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation
        if image_size is not None:
            self.image_size = image_size
        else:
            self.image_size = None

    def __call__(self, img):
        h, w = img.size

        if self.image_size is not None:
            if h < self.image_size or w < self.image_size:
                return transforms.Resize(self.image_size, self.interpolation)(img)

        if h < self.size or w < self.size:
            return img
        else:
            return transforms.Resize(self.size, self.interpolation)(img)


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _preprocess1():
    return Compose([
        Resize((256, 256)),
        _convert_image_to_rgb,
        AdaptiveResize(768),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def _preprocess2():
    return Compose([
        Resize((256, 256)),
        _convert_image_to_rgb,
        AdaptiveResize(768),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

class emd_loss(torch.nn.Module):
    """
    Earth Mover Distance loss
    """
    def __init__(self, dist_r=2,
        use_l1loss=False, l1loss_coef=0.0):
        super(emd_loss, self).__init__()
        self.dist_r = dist_r
        self.use_l1loss = use_l1loss
        self.l1loss_coef = l1loss_coef

    def check_type_forward(self, in_types):
        assert len(in_types) == 2

        x_type, y_type = in_types
        assert x_type.size()[0] == y_type.shape[0]
        assert x_type.size()[0] > 0
        # assert len(x_type) == 2
        # assert len(y_type) == 2

    def forward(self, x, y_true):
        self.check_type_forward((x, y_true))
        if y_true.size()[1] == 5:
            coff = 1.0 - torch.sum(y_true.pow(2), dim=-1) + 0.2
        else:
            coff = 1.0 - torch.sum(y_true.pow(2), dim=-1) + 0.1
        cdf_x = torch.cumsum(x, dim=-1)
        cdf_ytrue = torch.cumsum(y_true, dim=-1)
        if self.dist_r == 2:
            samplewise_emd = torch.sqrt(torch.mean(torch.pow(cdf_ytrue - cdf_x, 2), dim=-1))
        else:
            samplewise_emd = torch.mean(torch.abs(cdf_ytrue - cdf_x), dim=-1)
        samplewise_emd = samplewise_emd.mul(coff)
        loss = torch.mean(samplewise_emd)
        if self.use_l1loss:
            rate_scale =  torch.Tensor([float(i+1) for i in range(x.size()[1])]).cuda()
            x_mean = torch.mean(x * rate_scale, dim=-1)
            y_true_mean = torch.mean(y_true * rate_scale, dim=-1)
            l1loss_coef = 1.0 - torch.abs(y_true_mean - 0.5)
            l1 = (x_mean - y_true_mean).pow(2)
            l1loss = torch.mean(l1.mul(l1loss_coef))
            loss += l1loss
        return loss