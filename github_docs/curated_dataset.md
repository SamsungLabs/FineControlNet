# Download our curated dataset

Download our curated dataset from [here](https://drive.google.com/file/d/1KcN9pMHQyLdOBVS2GuUdybyBk33rXgbx/view?usp=sharing). Or you could parse it yourself.


# Parse MSCOCO dataset

### Download MSCOCO 

- Download MSCOCO 2017 keypoints dataset's images and annotations from [here](https://github.com/jin-s13/COCO-WholeBody). 
- We only parse the validation set (~5K images), but you can also parse the train set (~200K images) for your purpose.

```
${path/to/datasets}
|   |-- MSCOCO  
|   |   |-- images  
|   |   |   |-- train2017  
|   |   |   |-- val2017  
|   |   |-- annotations  
|   |   |   |-- coco_wholebody_train_v1.0.json
|   |   |   |-- coco_wholebody_val_v1.0.json
|   |-- OtherPoseDataset1
|   |-- OtherPoseDataset2
|   |-- ...  
```


### Parse MSCOCO poses


Run below command.

```
cd ./data_generation
python3 parse_mscoco_poses.py --datadir {path/to/MSCOCO} --subset val --resolution 512
```


- The resolution argument is the same with the resolution of the ControlNet encoder's 2D input.
- The annotation file will be saved to `f'coco_{data_split}_pose_data_finecontrolnet.json'`.


### Annotate text prompts

Run below command.

```
python3 add_prompts.py
```



- Then, the final annotations will be save in `f'coco_{data_split}_pose_with_prompt_data_finecontrolnet.json'`
- To the prompts are assigned randomly from a list of person identities and scene settings. These can be changed/modified inside this script by modifying the instance_descs variable and/or the setting_descs variable.


### Annoation format

In the saved `f'coco_{data_split}_pose_with_prompt_data_finecontrolnet.json.json'` json file, data is structured as below:


```
'annotations': [
    {
        'id': annotation id,  # following MSCOCO format
        'image_id': image id, # following MSCOCO format
        'people': {
            'poses': [ # In OpenPose + ControlNet style. You can directly put this to utils.draw_pose function in the ControlNet's OpenPose code.
                {
                    'bodies': {
                        'candidate': (18, 3) list # 18 is the number of joints, x,y location normalized in 0~1, third element is 1 if valid 0 if not
                        'subset': (20) # np.arange(20). last two are dummy. -1 if the jth joint is not valid
                    },
                    'hands': [right hand joint list, left hand joint list], # [(25,2), (25,2)] x,y normalized in 0~1. if no hands, ann['people']['poses'][idx]['hands'] = []
                    'faces': [face_joint_list] # [68,2] x,y normalized in 0~1. if no hands, ann['people']['poses'][idx]['faces'] = []
                }, ...

            ], 
            'res_ratios': [], # list of bounding box resolutions of each pose
            'crowd_indices' : [] # list of CrowdIndex of each pose
        }
        'global_desc': str, # global description of the generated scene
        'instance_descs': [str, str, ...]  # instance level identity description
        'setting_desc': str, # description of the setting (ex. background)
        'seed': int, # random seed to generate the same image

    }, ...
],

'images': [
    {
        'id': ... # same with the original MSCOCO: https://cocodataset.org/#format-data
    }
]
```