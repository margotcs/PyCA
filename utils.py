import os
import torch
from PIL import Image
import numpy as np
import cv2

def launch_video(size, fps, fourcc='avc1'):
    """
    Returns Videowriter, ready to record and save the video.

    Parameters:
    size: (H,W) 2-uple size of the video
    fps: int, frames per second
    fourcc : Encoder, must work with .mp4 videos
    """
    os.makedirs('Videos', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*fourcc)
    numvids = len(os.listdir('Videos/'))
    vid_loc = f'Videos/Vid_{numvids}.mp4'
    return cv2.VideoWriter(vid_loc, fourcc, fps, (size[1], size[0]))

def add_frame(writer, worldmap):
    frame = worldmap.transpose(1,0,2)  # (H,W,3)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    writer.write(frame)

def save_image(worldmap):
    os.makedirs('Images', exist_ok=True)
    numimgs = len(os.listdir('Images/'))
    img_name = f'img_{numimgs}.png'
    save_tensor_image(torch.tensor(worldmap).permute(2, 1, 0), img_name)

def save_tensor_image(tensor, filename):
    """
    Save a tensor as an image file.

    Parameters:
    tensor: torch.Tensor, the tensor to save
    filename: str, the name of the file to save the image as
    """
    # Convert the tensor to a numpy array
    np_array = tensor.cpu().numpy()

    # Convert the numpy array to a PIL Image
    image = Image.fromarray(np_array.astype('uint8').transpose(1, 2, 0))

    # Save the image
    image.save(os.path.join('Images', filename))