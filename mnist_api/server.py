from fastapi import FastAPI
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch
from mnist_api.model import Net
    
model = Net()
model.load_state_dict(torch.load("mnist_api/mnist_model.pth"))
model.eval()

image = np.load('mnist_api/test_data.npy')

app = FastAPI()

@app.get('/')
def reed_root():
    return {'message': 'MNIST model API'}

@app.post('/predict')
def predict(input_data):
    '''
    Predicts the numner on the image

    Args:
        input_data: image representing the number

    Returns:
        int: the predicted number
    '''
    x = torch.from_numpy(image)
    x = x.unsqueeze(0)
    y = model(x)
    y = y.argmax(dim=1, keepdim=True).item()
    return y 