import torch
import numpy as np
from typing import List
from PIL import Image

import clip
from torchmetrics.multimodal import CLIPScore
from torchvision.transforms.functional import crop


class TextMetrics:
    def __init__(self,
                 device,
                 clip_similarity_score_model_name="openai/clip-vit-base-patch16"):
        self.device = device
        self.clip_global_similarity = CLIPScore(
            model_name_or_path=clip_similarity_score_model_name).to(self.device)

        self.clip_local_similarity = CLIPScore(
            model_name_or_path=clip_similarity_score_model_name).to(self.device)

        self.model, self.preprocess = clip.load('ViT-B/32', self.device)

    def compute_global_clip_score(self,
                                  global_text_prompts: List[str],
                                  pred_images: torch.tensor):
        """computes image-level score for similarity with the text prompt

        :param global_text_prompts: list of prompts (N, )
        :param pred_images: list of images (N, C, H, W)
        :return: similarity scores
        """
        if type(pred_images) is np.ndarray:
            pred_images = torch.tensor(pred_images).to(self.device)

        if pred_images.shape[-1] == 3:
            pred_images = pred_images.permute(0, 3, 1, 2)

        with torch.no_grad():
            clip_similarity_value = self.clip_global_similarity(pred_images, global_text_prompts)

        clip_similarity_result = {
            "global_clip": float(clip_similarity_value.cpu().numpy().mean()),
        }

        results = {**clip_similarity_result}

        return results

    def compute_local_clip_score(self,
                                 local_text_prompts: List[List[str]],
                                 bounding_boxes: List[List[int]],
                                 pred_images: torch.tensor,
                                 ):
        """computes instance-level score for similarity with the text prompt

        :param local_text_prompts: list of per-instance prompts
        :param pred_images: list of images (N, C, H, W)
        :param bounding_boxes: list of bounding boxes in [left, top, right, bottom] convention
        """

        if type(pred_images) is np.ndarray:
            pred_images = torch.tensor(pred_images).to(self.device)

        if pred_images.shape[-1] == 3:
            pred_images = pred_images.permute(0, 3, 1, 2)

        assert len(local_text_prompts) == len(bounding_boxes)

        results = []
        for ii in range(len(local_text_prompts)):
            results_per_image = []
            for jj in range(len(local_text_prompts[ii])):
                with torch.no_grad():
                    bbox = bounding_boxes[ii][jj]
                    cropped_image = crop(pred_images[ii], top=bbox[1], left=bbox[0],
                                         height=bbox[3] - bbox[1], width=bbox[2] - bbox[0])
                    clip_similarity_value = self.clip_local_similarity(cropped_image, local_text_prompts[ii][jj])
                    results_per_image.append(clip_similarity_value)

            image_local_clip_score = sum(results_per_image) / len(results_per_image)
            results.append(image_local_clip_score)

        results_mean = sum(results) / len(results)

        clip_similarity_result = {
            "local_clip": float(results_mean.cpu().numpy().mean()),
        }

        return {**clip_similarity_result}

    def compute_global_clip_score_openai(self,
                                         global_text_prompts: List[str],
                                         pred_image: Image.Image):

        image = self.preprocess(pred_image).unsqueeze(0).to(self.device)
        text = clip.tokenize(global_text_prompts).to(self.device)

        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(image, text)

        clip_similarity_result = {
            "openai_global_clip": float(logits_per_image.cpu().numpy().mean()),
        }

        results = {**clip_similarity_result}

        return results

    def compute_local_clip_score_openai(self,
                                        local_text_prompts: List[List[str]],
                                        bounding_boxes: List[List[int]],
                                        pred_image: Image.Image):

        results_logits = []
        results_normalized = []
        results_differences = []
        for ii in range(len(local_text_prompts)):
            results_per_image_logits = []
            results_per_image_normalized = []
            results_per_image_differences = []
            for jj in range(len(local_text_prompts[ii])):
                bbox = bounding_boxes[ii][jj]
                cropped_image = pred_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                image = self.preprocess(cropped_image).unsqueeze(0).to(self.device)
                instance_text = clip.tokenize([local_text_prompts[ii][jj]]).to(self.device)
                all_instances_text = clip.tokenize(local_text_prompts[ii]).to(self.device)
                with torch.no_grad():
                    logits_image, logits_text_instance = self.model(image, instance_text)
                    logits_image_all_instances, logits_text_all_instances = self.model(image, all_instances_text)

                    results_per_image_logits.append(logits_text_instance.cpu().numpy().item())
                    logits_image_all_instances = logits_image_all_instances[0]
                    clip_similarity_normalized = logits_image_all_instances.softmax(dim=-1).cpu().numpy()[jj]
                    results_per_image_normalized.append(clip_similarity_normalized)

                    target_logit = logits_image_all_instances[jj]
                    nontarget_logits = torch.cat((logits_image_all_instances[:jj], logits_image_all_instances[jj+1:]))
                    clip_similarity_differences = target_logit - nontarget_logits.mean()
                    results_per_image_differences.append(clip_similarity_differences.cpu().numpy())

            results_logits.append(np.mean(results_per_image_logits))
            results_normalized.append(np.mean(results_per_image_normalized))
            results_differences.append(np.mean(results_per_image_differences))

        clip_similarity_result = {
            "openai_local_clip_logits": float(np.mean(results_logits)),
            "openai_local_clip_normalized": float(np.mean(results_normalized)),
            "openai_local_clip_differences": float(np.mean(results_differences)),
        }

        return {**clip_similarity_result}
