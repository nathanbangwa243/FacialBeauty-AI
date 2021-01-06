import logging

import azure.functions as func

import os

import tensorflow as tf
import numpy as np


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('name')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        cwd = os.getcwd()

        listdir = os.listdir(cwd)

        return func.HttpResponse(f"cwd : {cwd}\nlistdir : {listdir}")
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )
