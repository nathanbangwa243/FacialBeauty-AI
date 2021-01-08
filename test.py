# HTPP Request
import requests

# Data visualization
from pprint import pprint

# Sysrem
import os
import sys

# Data
import json
import numpy as np


class APIsRequest:
    headers = {
        "x-functions-key": "mgA0ahcvaQarbuowqc9/ioA7isN3oUJ8Wa2/GLX13Pe3W6UgfFiukQ=="
    }

    LOCALHOST_ENDPOINT = "http://localhost:7071/api/FacialBeauty"
    AZURE_ENDPOINT = "https://facialbeauty.azurewebsites.net/api/FacialBeauty"

    def facialBeauty(self, imageFile=None, online=True):
        # assert that image exist
        assert os.path.isfile(imageFile)

        # switch endpoint
        endPoint = self.AZURE_ENDPOINT if online else self.LOCALHOST_ENDPOINT

        # read image stream
        with open(imageFile, 'rb') as fp:
            data = fp.read()

        # type data
        print(f"Data type : {type(data)}")

        # HTTP Parameters
        params = {}

        # HTTP Request
        response = requests.get(
            # endPoint
            url=endPoint,
            
            #headers  
            headers=self.headers,

            # params 
            params=params,

            # image stream 
            data=data
        )

        print("[RESPONSE]\n", response.json())


# instance
apisRequest = APIsRequest()

# image
imageFile = os.path.join(os.getcwd(), "images", "face_filter_ex.png")

apisRequest.facialBeauty(imageFile=imageFile, online=False)
