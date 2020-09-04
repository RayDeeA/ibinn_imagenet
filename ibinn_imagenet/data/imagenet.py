import numpy as np
import torch

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from torchvision.transforms import transforms as T
from torchvision.datasets.imagenet import ImageFolder

class UnNormalize:
    def __init__(self, mus, sigmas):
        self.m = mus
        self.s = sigmas

    def __call__(self, x):
        for i in range(x.shape[1]):
            x[:, i] = x[:, i] * self.s[i] + self.m[i]
        return x

class Imagenet():

    def __init__(self, root_folder_train, root_folder_val, batch_size, download = False):
        super().__init__()

        if download:
            import torchvision
            torchvision.datasets.ImageNet(root_folder_train, split="train", download=True)
            torchvision.datasets.ImageNet(root_folder_val, split="val", download=True)

        self.root_folder_train = root_folder_train
        self.root_folder_val = root_folder_val

        self.val_fast_set_size = 2000 # Use only 2000 randomly picked files for validation steps performed within epochs

        self.batch_size = batch_size

        self.n_classes = 1000
        self.img_crop_size = (224, 224)

        self._mu_img = [0.485, 0.456, 0.406]
        self._std_img = [0.229, 0.224, 0.225]

        self._all_one_hot_encodings = torch.eye(self.n_classes)

        self.train_data = ImageFolder(self.root_folder_train + "/train", transform=T.Compose([
            T.RandomResizedCrop(self.img_crop_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(self._uniform_noise),
            T.Normalize(self._mu_img, self._std_img)
        ]), target_transform=T.Compose([T.Lambda(self._class_to_soft_hot)]))

        self.val_data_fast = ImageFolder(self.root_folder_val + "/val", transform=T.Compose([
            T.Resize(256),
            T.CenterCrop(self.img_crop_size),
            T.ToTensor(),
            T.Normalize(self._mu_img, self._std_img)
        ]), target_transform=T.Compose([T.Lambda(self._class_to_one_hot)]))

        indices = np.random.choice(len(self.val_data_fast.imgs), self.val_fast_set_size).tolist()
        self.val_data_fast.imgs = [self.val_data_fast.imgs[i] for i in indices]
        self.val_data_fast.samples = self.val_data_fast.imgs
        self.val_data_fast.targets = [s[1] for s in self.val_data_fast.samples]

        self.val_data = ImageFolder(self.root_folder_val + "/val", transform=T.Compose([
            T.Resize(256),
            T.CenterCrop(self.img_crop_size),
            T.ToTensor(),
            T.Normalize(self._mu_img, self._std_img),
        ]), target_transform=T.Compose([T.Lambda(self._class_to_one_hot)]))

        self.val_data_10_crop = ImageFolder(self.root_folder_val + "/val", transform=T.Compose([
            T.Resize(256),
            T.TenCrop(self.img_crop_size),
            T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])),
            T.Lambda(lambda crops: torch.stack([T.Normalize(self._mu_img, self._std_img)(crop) for crop in crops])),
        ]), target_transform=T.Compose([T.Lambda(self._class_to_one_hot)]))

        self.train_loader =         torch.utils.data.DataLoader(self.train_data, batch_size=100, shuffle=True, num_workers=12, pin_memory=False, sampler=None)
        self.val_loader_fast =      torch.utils.data.DataLoader(self.val_data_fast, batch_size=10, shuffle=False, num_workers=12, pin_memory=False, sampler=None)
        self.val_loader =           torch.utils.data.DataLoader(self.val_data, batch_size=100, shuffle=True, num_workers=12, pin_memory=False, sampler=None)
        self.val_loader_10_crop =   torch.utils.data.DataLoader(self.val_data_10_crop, batch_size=10, shuffle=True, num_workers=12, pin_memory=False, sampler=None)

        self.unnormalize_im = UnNormalize(self._mu_img, self._std_img)

    def set_model(self, model):
        self.train_loader.set_model(model)

    def _uniform_noise(self, x):
        return torch.clamp(x + torch.rand_like(x) / 255.,  min=0., max=1.)

    def _class_to_soft_hot(self, y):
        hard_one_hot = self._all_one_hot_encodings[y]

        return hard_one_hot * (1 - 0.05) + 0.05 / self.n_classes

    def _class_to_one_hot(self, y):
        hard_one_hot = self._all_one_hot_encodings[y]

        return hard_one_hot




