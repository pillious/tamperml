import pandas as pd
import os
import csv
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
local_path = os.getenv('DATASET_PATH')

table_paths = []

tp_list = os.listdir(local_path + 'CASIA2.0_revised/Tp')
au_list = os.listdir(local_path + 'CASIA2.0_revised/Au')
gt_list = os.listdir(local_path + 'CASIA2.0_Groundtruth')

for tp_path in tp_list:
    parts = tp_path.replace('.', '_').split('_')[5:8]
    paths = []
    paths.append([s for s in au_list if f'{parts[0][:3]}_{parts[0][3:]}' in s][0])
    paths.append([s for s in gt_list if '_'.join(parts) in s][0])
    paths.append(tp_path)
    
    table_paths.append(paths)


with open('ids.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    header = ["source", "target", "tampered"]
    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(table_paths)

