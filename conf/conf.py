
import os
import sys
from datetime import datetime
from pytz import timezone, utc

def localTime(*args):
    utc_dt = utc.localize(datetime.utcnow())
    my_tz = timezone("Asia/Ho_Chi_Minh")
    converted = utc_dt.astimezone(my_tz)
    return converted.timetuple()

import logging
logging.basicConfig(
    format="""%(asctime)s %(processName)-10s %(name)s %(levelname)-8s - %(message)s""", 
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.WARNING
)

logging.Formatter.converter = localTime

local_config_dict = {
    "DB_MYSQL_HOST"                 : "localhost", #"host.docker.internal",
    "DB_MYSQL_PORT"                 : "3306",
    "DB_MYSQL_USERNAME"             : "root",
    "DB_MYSQL_PASSWORD"             : "",
    "DB_MYSQL_NAME"                 : "BIB_project",
    "DB_MYSQL_IMG_TABLE"            : "image_url",
    "DB_MYSQL_CANDIDATE_TABLE"      : "Candidate_Info",
    "DB_MYSQL_PREDICTION_TABLE"     : "BIB_prediction",
                                    
}

# Local initial
for key, value in local_config_dict.items():
    if os.environ.get(key) is None:
        os.environ[key] = value

MYSQL = {}
MYSQL['DB_MYSQL_HOST'] = os.environ['DB_MYSQL_HOST']
MYSQL['DB_MYSQL_PORT'] = os.environ['DB_MYSQL_PORT']
MYSQL['DB_MYSQL_USERNAME'] = os.environ['DB_MYSQL_USERNAME']
MYSQL['DB_MYSQL_PASSWORD'] = os.environ['DB_MYSQL_PASSWORD']
MYSQL['DB_MYSQL_NAME'] = os.environ['DB_MYSQL_NAME']
DB_MYSQL_IMG_TABLE = os.environ['DB_MYSQL_IMG_TABLE']
DB_MYSQL_CANDIDATE_TABLE = os.environ['DB_MYSQL_CANDIDATE_TABLE']
DB_MYSQL_PREDICTION_TABLE = os.environ['DB_MYSQL_PREDICTION_TABLE']

NUM_ROW = 2

def gpu_available():
    """
        Check NVIDIA with nvidia-smi command
        Returning code 0 if no error, it means NVIDIA is installed
        Other codes mean not installed
    """
    code = os.system('nvidia-smi')
    return code == 0

if gpu_available() == False:
    DEVICE = 'cpu'
else:
    DEVICE = 'gpu'
