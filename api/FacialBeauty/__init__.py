import logging
from os import name, stat
import os

import azure.functions as func

# ML Toolkits
import tensorflow as tf
import cv2
import numpy as np

# Json
import json

import uuid

from . import tools


# HTTP CODE
CODE400 = 400   # request failed
CODE412 = 412   # precondition failed

CODE500 = 500   # internal server error
CODE200 = 200   # success


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # keys infos
    # image data : image data stream
    imageData = req.get_body()  # tools.checkParameter(req=req, name='imageData')

    logging.info(f"[imageData Type]: {type(imageData)}")

    # check required elements
    condition = bool(imageData)

    if condition:

        response = dict()

        try:
            # image stream to numpy array
            image = tools.imageStreamToArray(imageData)

            # preprocess
            image = tools.preprocessImage(image)

            # make FKP inference
            fkpMap = tools.makeInference(image)

            # classify
            response = tools.FacialBeautyClassifier(fkpMap)

        except Exception as error:
            response = {
                "message": f"{error}"
            }

            response = json.dumps(response)

            return func.HttpResponse(
                response,
                status_code=CODE400
            )

        else:
            response = json.dumps(response)

            return func.HttpResponse(
                response,
                status_code=CODE200
            )

    else:
        response = {
            "message": f"No devMode or httpAction or container or uidElement parameter"
        }
        response = json.dumps(response)

        return func.HttpResponse(
            response,
            status_code=CODE400
        )
