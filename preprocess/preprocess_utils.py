import argparse
import distutils
import os
import zipfile

import requests

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def get_parser():
    def strtobool(v):
        return bool(distutils.util.strtobool(v))

    parser = argparse.ArgumentParser(
        description='Preprocessing of HAR datasets')


    # parser.add_argument('--working_directory', default='run',
    #                     help='directory containing datasets, trained models and training logs')
    # parser.add_argument('--config', default='sample_configs/self_har.json',
    #                     help='')
    #
    # parser.add_argument('--labelled_dataset_path', default='run/processed_datasets/motionsense_processed.pkl', type=str,
    #                     help='name of the labelled dataset for training and fine-tuning')
    # parser.add_argument('--unlabelled_dataset_path', default='run/processed_datasets/hhar_processed.pkl', type=str,
    #                     help='name of the unlabelled dataset to self-training and self-supervised training, ignored if only supervised training is performed.')
    #
    # parser.add_argument('--window_size', default=400, type=int,
    #                     help='the size of the sliding window')
    # parser.add_argument('--max_unlabelled_windows', default=40000, type=int,
    #                     help='')
    #
    # parser.add_argument('--use_tensor_board_logging', default=True, type=strtobool,
    #                     help='')
    # parser.add_argument('--verbose', default=1, type=int,
    #                     help='verbosity level')

    return parser



def download_and_extract(files, links, data_dir):
    os.makedirs('../datasets/download', exist_ok=True)
    os.makedirs('../datasets/extracted', exist_ok=True)

    os.makedirs(data_dir, exist_ok=True)
    for i in range(len(files)):
        data_directory = os.path.abspath(data_dir+"/download/" + str(files[i]) + ".zip")
        if not os.path.exists(data_directory):
            print("downloading " + str(files[i]))
            download_url(links[i], data_directory)  # download the data
            # when that finishes, extract the data
            print("download done")
            with zipfile.ZipFile(data_directory, 'r') as zip_ref:
                zip_ref.extractall(data_dir + "/extracted/")
            print("extract done")
        else:
            print(str(files[i]) + "data already exists, skipping download")


