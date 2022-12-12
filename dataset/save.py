import torch
from tqdm import tqdm
import joblib, os

from config import *
from utils import merge_dataset, train_valid_processed_data, custom_clustering_data
from .data_processor import process_conell_data, process_idiris_dir, process_idiris_files
from .flair_dataset import FlairDataset


class SaveDataset(object):
    """
    Due to IPR we are not disclosing this part of the code.
    """
    pass