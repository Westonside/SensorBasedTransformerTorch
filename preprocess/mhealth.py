import numpy as np
import requests

from preprocess.preprocess_utils import download_and_extract

link = 'http://archive.ics.uci.edu/static/public/319/mhealth+dataset.zip'
download_and_extract(['MHEALTH'], [link], '../datasets/')
