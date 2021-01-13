import numpy as np

BATCH_SIZE          = 128
IMG_SHAPE  = 96  # Our training data consists of images with width of 150 pixels and height of 150 pixels
VALIDATION_SPLIT    = 0.20
EPOCHS              = 300
PATIENCE            = 30

TARGET_FKPS = np.array([49, 55, 28, 32, 36, 4, 14]) - 1
