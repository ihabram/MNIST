import numpy as np
from PIL import Image

def resize_28x28(input_image):
    # Convert image to grayscale
    img = input_image.convert('L')
    
    # Resize the image to 28x28 pixels
    img_resized = img.resize((28, 28), Image.LANCZOS)
    
    # Convert the image to a NumPy array
    img_array = np.array(img_resized, dtype=np.float32)

    # Negate the image
    img_array = 255 - img_array
    
    return img_array