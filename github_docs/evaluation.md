# Evaluation

To evaluate FineControlNet we use three classes of metrics for measuring 1) the quality of generated images, 2) text-image consistency between the generated image and input text prompts, and 3) pose control accuracy.

We assume that the generated images of FineControlNet using the dataset file `f'coco_{data_split}_pose_with_prompt_data_finecontrolnet.json'` are saved to a folder `/path/to/finecontrolnet_results`.

```
FINECONTROLNET_RESULTS="path/to/finecontrolnet_results"
```


The evaluation process is borrowed from [HumanSD](https://github.com/IDEA-Research/HumanSD/tree/main) and modified. Great thanks to the contributors!

## Environment setup
To run the evaluations, make sure that your python environment has the [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/) and [clip](https://github.com/openai/CLIP) packages installed:
```
pip install -r evaluation_requirements.txt
```

In addition, you will need to install [MMPose](https://github.com/open-mmlab/mmpose) for running the pose evaluations. Follow the instructions [here](https://mmpose.readthedocs.io/en/latest/installation.html) for installation. Make sure to install `mmpose==0.29.0`, and its corresponding compatible packages `mmdet 2.x` and `mmcv 1.x`.

What we did:

```
pip install --upgrade pip
pip install -U openmim
mim install "mmpose==0.29.0" "mmdet<3" "mmcv<2"
```



## Setup for computing FID and pose accuracy

__FID__: First download the reference dataset, HumanArt, from [here](https://idea-research.github.io/HumanArt/). And set the paths as below in `${ROOT}/metrics/metrics.yaml`.
```
quality:
    refer_dataset_base_dir: "/path/to/HumanArt"
    refer_dataset_json_path: "/path/to/HumanArt/HumanArt/annotations/validation_humanart.json"
```

__pose accuracy__: Download the 2D pose estimator's checkpoint `higherhrnet_w48_humanart_512x512_udp.pth` trained on HumanArt from [here](https://drive.google.com/drive/folders/1NLQAlF7i0zjEpd-XY0EcVw9iXP5bB5BJ) and place it under `${ROOT}/metrics/humansd_data`.


## Running the evaluations

For the image quality metric, we report the FID scores. To evaluate FineControlNet on the image quality metrics, run:
```
python -m metrics.evaluation_metrics \ 
--results_dir $FINECONTROLNET_RESULTS \ 
--method_name finecontrolnet \ 
--dataset_path ./data_generation/coco_{data_split}_pose_with_prompt_data_finecontrolnet.json \ 
--quality
```

For the text-image consistency evaluation, we report a set of CLIP-based similarity scores at the instance level. To evaluate FineControlNet on the image-text consistency metrics, run:
```
python -m metrics.evaluation_metrics \ 
--results_dir $FINECONTROLNET_RESULTS \ 
--method_name finecontrolnet \ 
--dataset_path ./data_generation/coco_{data_split}_pose_with_prompt_data_finecontrolnet.json \ 
--text
```

For evaluating the performance on the pose control accuracy metrics, run:
```
python -m metrics.evaluation_metrics \ 
--results_dir $FINECONTROLNET_RESULTS \ 
--method_name finecontrolnet \ 
--dataset_path ./data_generation/coco_{data_split}_pose_with_prompt_data_finecontrolnet.json \ 
--pose
```

> If you have error regarding xcocotools when running pose metric computation, please upgrade numpy  
> If you upgrade numpy, you could have an error regarding mmpose about np.int  
> Change np.int to int in L1159 and 1202 of the file "/path/to/python3.8/site-packages/mmpose/datasets/pipelines/bottom_up_transform.py"