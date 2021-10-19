import time
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import os
import argparse
import logging
from src.utils.all_utils import read_yaml, create_directory

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, 'running_logs.log'), level=logging.INFO, format=logging_str,
                    filemode="a")

def train_model(config_path,params_path):
    config=read_yaml(config_path)
    params=read_yaml(params_path)
    artifacts_dir=config["artifacts"]["ARTIFACTS_DIR"]
    ckpt_dir=config["artifacts"]["model_ckpt_dir"]
    ckpt_path=config["artifacts"]["model_ckpt_path"]
    base_model_dir=config["artifacts"]["BASE_MODEL_DIR"]
    updated_base_model_name=config["artifacts"]["UPDATED_BASE_MODEL_NAME"]
    base_model_dir_path=os.path.join(artifacts_dir,base_model_dir)
    updated_base_model_path=os.path.join(base_model_dir_path,updated_base_model_name)
    batch_size=params["BATCH_SIZE"]
    epoch=params["EPOCHS"]
    ckpt_dir_path=os.path.join(artifacts_dir,ckpt_dir)
    create_directory([ckpt_dir_path])
    CKPT_path=os.path.join(ckpt_dir_path,ckpt_path)


    def get_log_path(log_dir="logs/fit"):
         uniqueName = time.strftime("log_%Y_%m_%d_%H_%M_%S")
         log_path = os.path.join(log_dir, uniqueName)
         print(f"savings logs at: {log_path}")

         return log_path

    train=ImageDataGenerator(rescale=1/255,rotation_range=40,
         width_shift_range=0.2,
         height_shift_range=0.2,
         shear_range=0.2,
         zoom_range=0.2,
         horizontal_flip=True,
         fill_mode='nearest')

    validation = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    train_data=train.flow_from_directory("C:/Users/kisho/Downloads/deep_learning_p1/scratch_detection/artifacts/data/train/",target_size=(224,224),batch_size=batch_size,class_mode="binary")
    validation_data=validation.flow_from_directory("C:/Users/kisho/Downloads/deep_learning_p1/scratch_detection/artifacts/data/val/",target_size=(224,224),batch_size=batch_size,class_mode="binary")
    
    log_dir = get_log_path()
    file_writer = tf.summary.create_file_writer(logdir=log_dir)
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_path, save_best_only=True)
    CALLBACKS_LIST = [tensorboard_cb, early_stopping_cb, checkpointing_cb]
    model=tf.keras.models.load_model(
    updated_base_model_path, custom_objects=None, compile=True, options=None)
    model.fit_generator(
        train_data,
        validation_data=validation_data,
        epochs=epoch,callbacks=CALLBACKS_LIST ,verbose=1)



if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument("--config", "-c", default="config/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")

    parsed_args = args.parse_args()

    try:
        logging.info(">>>>> stage three started")
        train_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info("stage three completed!  model trained >>>>>")
    except Exception as e:
        logging.exception(e)
        raise e