import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

if __name__ == "__main__":

    # Load modal
    model = tf.keras.models.load_model("model/unet (10).h5")

    # Read image contain person
    original = cv2.imread('dress_.jpg')
    heightPersonImage, widthPersonImage, _ = original.shape   # Image: 3D array

    # Background
    background = cv2.imread('background.jpg')
    background = cv2.resize(background, (widthPersonImage, heightPersonImage))

    # Tien hanh predict mask
    img = cv2.resize(original, (224, 224))
    img = img/255.0
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    pred_mask = model.predict(img)[0] 

    # Flatten 3D array to 2D array
    pred_mask = (pred_mask > 0.5) * 255
    pred_mask = pred_mask.astype(np.uint8)
    pred_mask = cv2.resize(pred_mask, (widthPersonImage, heightPersonImage))

    # get not of mask
    pred_mask_inv = cv2.bitwise_not(pred_mask)

    # Now black-out the area of logo in ROI
    print(pred_mask_inv.shape)
    img1_bg = cv2.bitwise_and(background,background,mask = pred_mask_inv)


    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(original,original,mask = pred_mask)

    # Put logo in ROI and modify the main image
    background = cv2.add(img1_bg,img2_fg)

    cv2.imshow("mask", background)
    cv2.waitKey(0)
    exit(0)
    
