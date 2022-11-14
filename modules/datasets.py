"""Datasets
"""

from torch.utils.data import Dataset
import albumentations as A
import numpy as np
import cv2
import os
import random

class SegDataset(Dataset):
    """Dataset for image segmentation

    Attributs:
        x_dirs(list): 이미지 경로
        y_dirs(list): 마스크 이미지 경로
        input_size(list, tuple): 이미지 크기(width, height)
        scaler(obj): 이미지 스케일러 함수
        logger(obj): 로거 객체
        verbose(bool): 세부 로깅 여부
    """   
    def __init__(self, paths, input_size, scaler, mode='train', logger=None, verbose=False):
        
        self.x_paths = paths
        self.y_paths = list(map(lambda x : x.replace('x', 'y'),self.x_paths))
        self.input_size = input_size
        self.scaler = scaler
        self.logger = logger
        self.verbose = verbose
        self.mode = mode
        self.shadower = A.ColorJitter(brightness=[0.2, 0.5], saturation=[1.3, 1.5], always_apply=True)

    def get_transform(self, x):
        if self.mode == "train":
            return self.get_train_transform(x)
        elif self.mode == "valid":
            return self.get_val_transform(x)
        else:
            return self.get_test_transform(x)

    def get_train_transform(self, x):
        """generate transform instance for train"""

        # datas = dict(xs, **dict(masks=ys))
        image_transform = A.Compose(
            [
                # A.HorizontalFlip(p=0.2),
                # ToTensorV2(transpose_mask=True),
                A.OneOf([
                    # A.VerticalFlip(p=0.2),
                    A.GaussNoise(p=1),
                    A.Blur(p=1),
                    A.Sharpen(p=1),
                    A.ISONoise(p=1),
                    A.MotionBlur(p=1),
                    A.IAAEmboss(),
                    A.CLAHE(),
                    # A.RandomBrightnessContrast(p=1),  # here
                    A.NoOp(p=1),
                ], p=1),

            ],
            # A.HorizontalFlip(p=0.2),
        )
        transform_data = image_transform(image=x)
        return transform_data["image"]

    def get_val_transform(self, x):
        """generate transform instance for train"""

        # datas = dict(xs, **dict(masks=ys))
        # common_transform = A.Compose(
        #     [
        #         # ToTensorV2(transpose_mask=True),
        #     ],
        #     additional_targets=additional_targets,
        # )
        # transform_datas = common_transform(**datas)

        return x

    def get_test_transform(self, xs, ys):
        """generate transform instance for train"""

        return xs, ys
    
    def shadow(self, x):
        h, w, c = x.shape
        left, right = x[:, :w//2, :], x[:, w//2:, :]
        trans = self.shadower(image=left)
        cat_x = np.concatenate((left, trans["image"]), axis=1)

        assert cat_x.shape == (384,736, 3), f"{cat_x.shape}"

        return cat_x
    
    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, id_: int):
        
        filename = os.path.basename(self.x_paths[id_]) # Get filename for logging
        additional_targets = None

        if self.mode == "train":
            x = cv2.imread(self.x_paths[id_], cv2.IMREAD_COLOR)
            orig_size = x.shape
            x = cv2.resize(x, self.input_size)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

            y = cv2.imread(self.y_paths[id_], cv2.IMREAD_GRAYSCALE)
            y = cv2.resize(y, self.input_size, interpolation=cv2.INTER_NEAREST)

            if random.random() > 0.1:
                x = self.get_transform(x)
            else:
                x = self.shadow(x)
                y = np.zeros_like(y)

            x = x.astype("float")
            x /= 255.
            x = np.transpose(x, (2, 0, 1))

            # if additional_targets is None:
            #     xs = [transform_data["image"]]
            # else:
            #     xs = [transform_data["image"], transform_data["image2"]]

            # ys = transform_data["masks"]

            return x, y, filename

        elif self.mode == "valid":
            x = cv2.imread(self.x_paths[id_], cv2.IMREAD_COLOR)
            orig_size = x.shape
            x = cv2.resize(x, self.input_size)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = x.astype("float")
            x /= 255.
            # x = self.scaler(x)
            x = np.transpose(x, (2, 0, 1))

            y = cv2.imread(self.y_paths[id_], cv2.IMREAD_GRAYSCALE)
            y = cv2.resize(y, self.input_size, interpolation=cv2.INTER_NEAREST)

            return x, y, filename
        
        elif self.mode == "test":
            x = cv2.imread(self.x_paths[id_], cv2.IMREAD_COLOR)
            orig_size = x.shape
            x = cv2.resize(x, self.input_size)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = x.astype("float")
            x /= 255.
            # x = self.scaler(x)
            x = np.transpose(x, (2, 0, 1))

            return x, orig_size, filename

        else:
            assert False, f"Invalid mode : {self.mode}"


