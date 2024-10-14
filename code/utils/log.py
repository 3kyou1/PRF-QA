import datetime
import os
import json
import logging

# Load config 
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)



log_file = 'log.log'
log_level = logging.DEBUG


log_format = '%(asctime)s - %(message)s'
date_format = '%Y-%m-%d %H:%M:%S'


logger = logging.getLogger('data_logger')
logger.setLevel(log_level)


file_handler = logging.FileHandler(log_file)
file_handler.setLevel(log_level)


formatter = logging.Formatter(log_format, date_format)
file_handler.setFormatter(formatter)


logger.addHandler(file_handler)


def generate_log(input):
    current_time = datetime.datetime.now().strftime(date_format)
    logger.info(f"[{current_time}]\n{input}")

def clear_log():
    if os.path.exists(log_file):
        os.remove(log_file)

def generate_tmp_log(input, log_file, log_level):
    logger = logging.getLogger('data_logger')
    logger.setLevel(log_level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    log_format = '%(asctime)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    current_time = datetime.datetime.now().strftime(date_format)
    logger.info(f"[{current_time}]\n{input}")


if __name__ == "__main__":
    clear_log()