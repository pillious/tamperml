import pandas as pd
import os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
local_path = os.getenv('DATASET_PATH')
print(local_path)

def read_csv(file):
    df = pd.read_csv(file)
    # print(file.head())
    img = Image.open("real_vs_fake/real-vs-fake/" + df['path'][0])
    img.show()

read_csv('/Users/timvuong/Desktop/hackWeek/archive/train.csv')
