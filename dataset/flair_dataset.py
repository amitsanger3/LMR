from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings
from sklearn import preprocessing
import torch
import time, os, json
import traceback

from config import pos_tagger, pos_dict_path, device, distance


class FlairDataset(object):
    """
    Due to IPR we are not disclosing this part of the code.
    """
    pass


class FlairDetection(FlairDataset):
    """
    Due to IPR we are not disclosing this part of the code.
    """
    pass
