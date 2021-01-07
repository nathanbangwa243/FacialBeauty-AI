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


# image type
IMAGE_TYPE = 'png'


def moderateImage(imageData):
    """
        Azure content moderator in action
        :return: tuple
        (status: bool, labelStat: dict)
    """

    status = False
    labelStat = {}

    # moderate image
    try:
        evaluateEndpoint = f"{config.CONTENT_MODERATOR_ENDPOINT}/contentmoderator/moderate/v1.0/ProcessImage/Evaluate"

        headers = {
            # Request headers
            'Content-Type': 'image/png',
            'Ocp-Apim-Subscription-Key': config.CONTENT_MODERATOR_SUBSCRIPTION_KEY,
        }

        params = {
            "overload": "stream"
        }

        response = requests.post(evaluateEndpoint,
                                 headers=headers, params=params, data=imageData)

        response = response.json()

        logging.info(f"Content Moderator: {response}")

        # check adult
        if response['IsImageAdultClassified']:
            return False, response

        else:
            return True, response

    except Exception as error:
        status = False
        labelStat = {"error": error}

    return status, labelStat


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # keys infos

    # dev mode : test or prod
    devMode = tools.checkParameter(req=req, name='devMode')

    # HTTP ACTION
    # httpAction = tools.checkParameter(req=req, name='httpAction')

    # container : USERS, SHOPS
    container = tools.checkParameter(req=req, name='container')

    # id target
    uidElement = tools.checkParameter(req=req, name='uidElement')

    # image data : image data stream
    imageData = req.get_body()  # tools.checkParameter(req=req, name='imageData')

    # file type
    # imageType = tools.checkParameter(req=req, name='imageType')

    logging.info(f"req.param: {req.params}")
    logging.info(f"req.files: {req.files}")
    logging.info(f"type req.get_body: {type(req.get_body())}")
    logging.info(f"req.get_body: {req.get_body()}")
    logging.info(f"devMode: {devMode}")
    # logging.info(f"httpAction: {httpAction}")
    # logging.info(f"imageData: {imageData}")

    # check required elements
    condition = bool(devMode) and \
        bool(container) and bool(imageData) and bool(
            uidElement)  # and bool(timestamp)

    if condition:

        # check container
        if isinstance(container, str):
            # to uppercase
            container = container.upper()

            if container not in ['USERS', 'SHOPS']:
                # Invalid container
                response = {
                    "message": f"Invalid container. Must be in ['USERS', 'SHOPS']"}
                response = json.dumps(response)

                return func.HttpResponse(
                    response,
                    status_code=CODE400
                )

            else:
                pass

        else:
            # Type error
            response = {
                "message": f"Invalid container type. Must be 'String'"}
            response = json.dumps(response)

            return func.HttpResponse(
                response,
                status_code=CODE400
            )

        try:
            # Moderate image
            isModerated, labelStat = moderateImage(imageData)

            if isModerated:
                # Azure Storage Connexion

                connexionString = ""

                # STORAGE
                if devMode == 'TEST':
                    connexionString = config.TEST_STORAGE_CONNEXION_STRING

                elif devMode == 'PROD':
                    connexionString = config.PROD_STORAGE_CONNEXION_STRING

                else:
                    response = {
                        "message": f"Invalid devMode. must be 'TEST' or 'PROD'"}
                    response = json.dumps(response)

                    return func.HttpResponse(
                        response,
                        status_code=CODE400
                    )

                # create the BlobServiceClient
                blobServiceClient = BlobServiceClient.from_connection_string(
                    conn_str=connexionString
                )

                # create container if not exist
                try:
                    container = container.lower()
                    blobServiceClient.create_container(container)
                except:
                    pass

                # Blob path : uidElement/name
                name = f"{tools.getTimestamp()}.{IMAGE_TYPE}"

                blobFile = os.path.join(uidElement, name)

                # blob client
                blobClient = blobServiceClient.get_blob_client(
                    container=container, blob=blobFile
                )

                # upload file to azure storage
                response = blobClient.upload_blob(imageData)

                # format response
                response = {
                    "imageURL": blobClient.url
                }

                response = json.dumps(response)

                return func.HttpResponse(
                    response,
                    status_code=CODE200
                )

            else:
                # no moderated content
                response = {
                    "message": f"Image is not moderated.", "labelStat": labelStat}

                response = json.dumps(response)

                return func.HttpResponse(
                    response,
                    status_code=CODE400
                )

        except Exception as error:
            response = {
                "message": f"{error}"}

            response = json.dumps(response)

            return func.HttpResponse(
                response,
                status_code=CODE400
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
