import json
import glob
import os
import os.path as osp
import numpy as np

if __name__ == '__main__':
    summary_metric = 'text' #'pose' #'quality/without_normalization'

    result_dir = f'/labdata/selim/finecontrol_baselines/comparisons/{summary_metric}'

    result_files = sorted(glob.glob(osp.join(result_dir, '*.json')))

    for rf in result_files:
        methodname = osp.basename(rf).split('_')[0]
        print()
        print("Method:  ", methodname)

        with open(rf) as f:
            data = json.load(f)
        for k, v in data.items():
            if 'Human Number' in k or summary_metric == 'text':
                print(k, np.mean(v), np.std(v))
            else:
                print(k, np.mean(v))
