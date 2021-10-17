from src.utils.all_utils import read_yaml,create_directory
import argparse
import pandas as pd
import os
from tqdm import tqdm
import logging
import splitfolders


logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'), level=logging.INFO, format=logging_str,
                    filemode="a")

def split_data(config_path):
    config=read_yaml(config_path)
    artifact_dir=config["artifacts"]["ARTIFACTS_DIR"]
    data=config["artifacts"]["data"]
    data_path=config["local_data_set_path"]
    create_directory([artifact_dir])
    splitfolders.ratio(data_path, output=os.path.join(artifact_dir,data), seed=68, ratio=(0.8, 0.2, 0.0), group_prefix=None)

    





if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info(">>>>> stage one started")
        split_data(config_path=parsed_args.config)
        logging.info("stage one completed! all the data are saved in local >>>>>")
    except Exception as e:
        logging.exception(e)
        raise e