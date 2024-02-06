import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.utilities.data import dim_zero_cat

from PIL import Image
import torchvision.transforms as TF
from tqdm import tqdm
import json
import os


class QualityMetrics():
    def __init__(self,
                 device,
                 refer_dataset_name: str = 'humanart',
                 refer_dataset_base_dir: str = '/labdata/datasets/HumanArt',
                 refer_dataset_json_path: str = '/labdata/datasets/HumanArt/HumanArt/annotations/validation_humanart.json',
                 fid_model_feature: int = 64,
                 kid_subset_size: int = 1000,
                 normalized: bool = True):

        # FID
        self.refer_dataset_base_dir = refer_dataset_base_dir
        self.refer_dataset_json_path = refer_dataset_json_path
        self.device = 'cpu'
        self.normalized = normalized

        
        if normalized:
            print("Normalize to 0~1")
            self.image_transforms = TF.Compose([
                TF.Resize(299),
                TF.CenterCrop(299),
                TF.ToTensor(),
            ])
        else:
            print("No normalization; 0~255")
            self.image_transforms = TF.Compose([
                TF.Resize(299),
                TF.CenterCrop(299),
                TF.PILToTensor(),
            ])
        dataset_imgs = []
        with open(refer_dataset_json_path, "r") as f:
            dataset_json = json.load(f)

        print("Initializing the reference dataset")
        for image_i in tqdm(range(len(dataset_json["images"]))):
            file_name = dataset_json["images"][image_i]["file_name"]
            # remove the first directory 'coco'
            if refer_dataset_name == 'coco':
                file_name = os.path.join('images', *file_name.split('/')[1:])
            present_image_path = os.path.join(refer_dataset_base_dir,file_name)
            img = Image.open(present_image_path).convert('RGB')
            dataset_imgs.append(self.image_transforms(img).unsqueeze(0))


        dataset_imgs = torch.concat(dataset_imgs).to(self.device)

        # FID
        print("Updating the FID model")
        self.fid_model_feature = fid_model_feature
        self.fid_model = FrechetInceptionDistance(feature=self.fid_model_feature, normalize=normalized).to(self.device) 
        self.fid_model.update(dataset_imgs, real=True)

        # KID
        print("Updating the KID model")
        self.kid_subset_size = kid_subset_size
        self.kid_model = KernelInceptionDistance(subset_size=self.kid_subset_size, normalize=normalized).to(self.device)
        self.kid_model.update(dataset_imgs, real=True)

    def calculate_fid(self, img):
        self.fid_model.update(img, real=False)
        return self.fid_model.compute()

    def calculate_kid(self, img):
        self.kid_model.update(img, real=False)
        if dim_zero_cat(self.kid_model.fake_features).shape[0] <= self.kid_model.subset_size:
            print(f'[INFO] More than {self.kid_model.subset_size} images needed when computing KID!')
            return None, None
        return self.kid_model.compute()

    def compute(self, pred_images):
        print('Computing FID and KID metrics...')
        if type(pred_images) is np.ndarray:
            pred_images = torch.tensor(pred_images)

        pred_images = pred_images.to(self.device)

        if pred_images.shape[-1] == 3:
            pred_images = pred_images.permute(0, 3, 1, 2)

        with torch.no_grad():
            fid_value = self.calculate_fid(pred_images)
            kid_value = self.calculate_kid(pred_images)

        fid_result = {
            "fid": fid_value.cpu().numpy().item(),
        }

        kid_result = {
            "kid": kid_value[0].cpu().numpy().item(),
        }

        results = {**fid_result, **kid_result}

        return results
