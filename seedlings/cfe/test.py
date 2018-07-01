import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms

from model import HybridNet

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

df_test = pd.read_csv('sample_submission.csv')

transform = transforms.Compose([
    transforms.RandomResizedCrop(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
        ])

label_map = {
    "Black-grass"               :0,
    "Charlock"                  :1,
    "Cleavers"                  :2,
    "Common Chickweed"          :3,
    "Common wheat"              :4,
    "Fat Hen"                   :5,
    "Loose Silky-bent"          :6,
    "Maize"                     :7,
    "Scentless Mayweed"         :8,
    "Shepherds Purse"           :9,
    "Small-flowered Cranesbill" :10,
    "Sugar beet"                :11}

model = HybridNet(12)
model.load_state_dict(torch.load('weights.pt'))
model.to(device)
model.eval()

predictions = []

print('predicting...')
for filename, category in tqdm(df_test.values, miniters=100):
    file_path = os.path.join('./data/test/', filename)
    img = Image.open(file_path)
    img_tensor = transform(img).unsqueeze_(0)
    tensor_input = img_tensor.to(device)

    output = model(tensor_input)
    output = output.cpu().detach().numpy()

    pred = np.argmax(output)

    print(tensor_input)
    print('file:', file_path, 'output:', output, 'label:', pred)

    predictions.append(list(label_map.keys())[list(label_map.values()).index(pred)])

df_test['species'] = predictions
df_test.to_csv('submission.csv', index=False)
