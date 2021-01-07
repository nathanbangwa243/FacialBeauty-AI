
from api.FacialBeauty import IMAGE_TYPE
import logging

import json

import os

import datetime

import numpy as np

import tensorflow as tf

import cv2
import io
from PIL import Image


# CONFIG
OPENCV_FKP_MODEL = ""


# Target FKP Model
FKP_MODELS = np.array([49, 55, 28, 32, 36, 4, 14])

# MOUTH FKP
LEFT_MOUTH_FKP      = 49
RIGHT_MOUTH_FKP     = 55

# NOSE FKP
PEEK_NOSE_FKP       = 28
LEFTBASE_NOSE_FKP   = 32
RIGHTBASE_NOSE_FKP  = 36

# EDGES FKP
LEFT_EDGE_FKP       = 4
RIGHT_EDGE_FKP      = 14


# IMAGE 
IMG_SHAPE = 96


# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, image:np.array):

        imageCopy = np.copy(image)

        # convert image to grayscale
        imageCopy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [0, 1]
        imageCopy=  imageCopy/255.0

        return imageCopy

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        # output_size est de type int ou tuple
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size # (h, w)

        new_h, new_w = int(new_h), int(new_w)

        newImage = cv2.resize(image, (new_w, new_h))

        return newImage


class AddBatchChannel(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # make a copy
        imageCopy = np.copy(image)

        # if image has no grayscale color channel, add one
        if(len(imageCopy.shape) == 2):
            # add that third color dim
            imageCopy = imageCopy.reshape(imageCopy.shape[0], imageCopy.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # batch image: C X H X W
        # image = image.transpose((2, 0, 1))
        
        return image


def imageStreamToArray(imageStream)->np.array:
    """Convert image stream to numpy array

    Args:
        imageStream (Bytes): image stream

    Returns:
        np.array: image matrix
    """

    # decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

    # (H, W, C)
    image = np.array(Image.open(io.BytesIO(imageStream)))

    return image


# model loader
def loadFKPModel(fkp:int) -> tf.keras.models.Sequential:
    """Model Loader

    Args:
        fkp (int): target fkp

    Returns:
        tf.keras.models.Sequential: Keras model
    """

    # Function project path
    cwd = os.getcwd()

    targetFile = os.path.join(cwd, "models", f"FKP{fkp}.h5")

    loadedModel = None

    try:
        # assert that is file
        assert os.path.isfile(targetFile)
        
        loadedModel = tf.keras.models.load_model(targetFile)

    except Exception as error:
        logging.info(f"[loadFKPModel Error] {error}")

        loadedModel = None
    
    return loadedModel


def preprocessImage(image:np.array)->np.array:
    """Image Preprocessing

    Args:
        image (np.array): image matrix

    Returns:
        np.array: image matrix
    """
    # copy image
    imageClone = np.copy(image)

    # RESCALE
    imageClone = Rescale((IMG_SHAPE, IMG_SHAPE))(imageClone)


    # NORMALIZATION
    imageClone = Normalize()(imageClone)

    # ADD BATCH CHANNEL
    imageClone = AddBatchChannel()(imageClone)

    return imageClone


def computeDistance(point1:np.array, point2:np.array)->np.float:
    """Compute distance between two points

    Args:
        point1 (np.array): First  point (x1,y1)
        point2 (np.array): Second point (x2, y2)

    Returns:
        np.float: distance = np.sqrt(())
    """

    # x and y distance
    xDistance = np.math.pow(point1[0] - point2[0], 2)
    yDistance = np.math.pow(point1[1] - point2[1], 2)

    # distance
    distance = np.math.sqrt(xDistance + yDistance)

    # convert to float
    distance = np.float(distance)

    return distance

def FacialBeautyClassifier(fkpMap:dict)->dict:
    """How much beauty is a face

    Args:
        fkpMap (dict): Facial Keypoints Map

    Returns:
        dict: facial beauty infos
        {
            # timestamp 
            "timestamp"             : float
            # MOUTH FKP
            "leftMouthFKP"          : FKP49
            "rightMouthFKP"         : FKP55
            "mouthWidth"            : distance(FKP49, FKP55)


            # NOSE FKP
            "peekNoseFKP"           : FKP28
            "leftBaseNoseFKP"       : FKP32
            "rightBaseNoseFKP"      : FKP36
            "noseWidth"             : distance(FKP32, FKP36)
            "peekLeftNose"         : distance(FKP28, FKP32)
            "peekRightNose"         : distance(FKP28, FKP36)


            # EDGE FKP
            "leftEdgeFKP"           : FKP4
            "rightEdgeFKP"          : FKP14
            
            # MOUTH-NOSE RATIO
            "MouthNoseAccuracy"     : mouthWidth / (noseWidth * 1.618)
            

            # MOUTH-EDGE RATIO
            "leftMouthEdgeInterval" : distance(FKP4, FKP49)
            "rightMouthEdgeInterval": distance(FKP55, FKP14)

            "leftMouthEdgeAccuracy" : mouthWidth / (leftMouthEdgeInterval * 1.618)
            "rightMouthEdgeAccuracy": mouthWidth / (rightMouthEdgeInterval * 1.618)

            "mouthEdgeAccuracy"     : float

            # NOSE RATIO
            "peekLeftNoseAccuracy"  : peekLeftNose  / (noseWidth * 1.618)
            "peekRightNoseAccuracy" : peekRightNose / (noseWidth * 1.618)

            "noseAccuracy"     : float
        }
    """

    pass


def makeInference(image:np.array)->dict:
    """Make prediction

    Args:
        image (np.array): Preprocess image matrix

    Returns:
        dict: facial keypoints
        {
            # fkp : [x,y]
        }
    """

    # FKP prediction map
    fkpMaps = {
        # fkp : [x,y]
    }

    # load models and predict

    for fkp in FKP_MODELS:
        # load model
        model = loadFKPModel(fkp)

        if not isinstance(model, None):
            # make prediction
            prediction = model.predict(image)

            # add fkp to fkpMaps
            fkpMaps[fkp] = prediction
        
        else:
            pass
    
    return fkpMaps


# function

def isNotNone(value):
    """Verifie si la valeur d'un element est pas null

    Args:
        value (Object): la valeur a verifier

    Returns:
        bool: True of False 
    """
    return not(value ==  None)
    

def checkParameter(req, name):
    # check parameter in body and req.params

    element = req.params.get(name)

    if element == None:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            # check apiKey and devMode in body
            element = req_body.get(name)
            logging.info(f"req_body: {req_body}")

    logging.info(
        f"[tools.checkParameter] {name} = {element}, type = {type(element)}")

    return element

# APIS

def getTimestamp(minutes=0):
    """
        Timestamp
    """
    now = datetime.datetime.now()

    now = now + datetime.timedelta(minutes=minutes)

    timestamp = datetime.datetime.timestamp(now)

    return timestamp


def getUTCTime(timestamp):
    """
        timestamp to utc time
    """

    utc = datetime.datetime.fromtimestamp(timestamp)

    utc = utc.isoformat()

    return utc
