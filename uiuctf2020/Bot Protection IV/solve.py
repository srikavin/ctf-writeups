import base64
import string
from io import BytesIO

import numpy as np
import requests
import torch
from PIL import Image
from bs4 import BeautifulSoup
from torchvision.transforms import ToTensor, Resize, Compose

mapping = {k: v for v, k in enumerate(string.ascii_uppercase)}
inv_mapping = {k: v for k, v in enumerate(string.ascii_uppercase)}


def get_label_string(label):
    batch = []
    for i in range(label.shape[0]):
        cur = []

        for row in range(label.shape[1]):
            max_idx = label[i][row].argmax()
            cur.append(inv_mapping[max_idx.item()])

        batch.append(''.join(cur))

    return batch


transform = Compose([Resize((224, 224)), ToTensor()])
model = torch.load('resnet_state').eval()


def get_prediction(image):
    image = transform(image).cuda().view(1, 3, 224, 224)
    model_output = model(image).view(-1, 5, 26)
    return get_label_string(model_output)


s = requests.Session()

while True:
    resp = s.get('https://captcha.chal.uiuc.tf/')

    soup = BeautifulSoup(resp.content, 'html.parser')

    captcha_src = soup.find_all("img")[1]['src'][len('data:image/png;base64,'):]

    im = BytesIO()
    im.write(base64.b64decode(captcha_src))

    image = Image.open(im)

    prediction = get_prediction(image)

    s.post('https://captcha.chal.uiuc.tf/', data={'captcha': prediction})

    print(soup.find_all("h2")[0], prediction, s.cookies)
