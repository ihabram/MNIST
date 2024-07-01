from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict
import numpy as np
import torch
from PIL import Image
import io

from mnist_api.model import Net
from mnist_api.preprocessor import resize_28x28
    
# Load the trained model
model = Net()
model.load_state_dict(torch.load("mnist_api/mnist_model.pth"))
model.eval()

# Global dictionary to store the uploaded image
uploaded_images: Dict[str, np.ndarray] = {}

app = FastAPI()

@app.get('/')
def root():
    return {'message': 'MNIST model API'}

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        #image_array = np.array(image, dtype=np.float32)
        uploaded_images["image"] = image
        return JSONResponse(content={"message": "Image uploaded successfully"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"An error occurred: {str(e)}")

@app.post('/predict')
def predict():
    '''
    Predicts the number represented on the image

    Returns:
        int: the predicted number
    '''
    if "image" not in uploaded_images:
        raise HTTPException(status_code=404, detail="No image found")
    
    image = uploaded_images["image"]
    processed_image = resize_28x28(image)

    x = torch.from_numpy(processed_image)
    x = x.unsqueeze(0)
    y = model(x)
    y = y.argmax(dim=1, keepdim=True).item()
    return y 