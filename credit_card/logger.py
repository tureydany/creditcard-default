import logging
from datetime import datetime
import os

Log_FILE_NAME=f"{datetime.now().strftime('%m%d%Y_%H%M%S')}.log"
Log_DIR_PATH=os.path.join(os.getcwd(),'logs')

Log_FILE_PATH=os.path.join(Log_DIR_PATH,Log_FILE_NAME)

logging.basicConfig(filename=Log_FILE_NAME,
format="[ %(asctime)s ] %(lineno)d %(name)s- %(levelname)s- %(message)s",
level=logging.INFO,)