import numpy as np
import warnings
warnings.filterwarnings("ignore")
import sys, shutil, os
import time, datetime
import traceback
import logging
logger = logging.getLogger(__name__)

# importing self dependencies
from config import *
from utils import *
from model import LMRBiLSTMAttnCRF

from test import Output

import torch
import torch.nn as nn


from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

torch.cuda.empty_cache()
torch.multiprocessing.set_start_method('spawn')


def run(model,
        trainable_params_info=True,
        loop=False
):

    print("Detection on files tweets starts...")
    try:
        if trainable_params_info:
            pytorch_total_params = sum(p.numel() for p in model.parameters())
            print("Total Params:", pytorch_total_params)

            pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("Total Trainable Params:", pytorch_total_params)

        if loop:
            n = 0
            while True:
                print(f"{n} Checking path: {geo_path} >>> {os.path.exists(geo_path)}")
                try:
                    if os.path.exists(geo_path):
                        file_processor = Output(model=model)
                        file_processor.process(filepath=geo_path)
                        try:
                            os.remove(geo_path)
                            print(f"{geo_path} -- removed 1")
                        except:
                            try:
                                op_path = f"/opt/LMR/{datetime.datetime.now().strftime('%d-%m-%Y_%H_%M_%S')}dump"
                                if not os.path.exists(op_path):
                                    os.makedirs(op_path)
                                shutil.move(geo_path, os.path.join(op_path, 'input.jsonl'))
                                print(f"{geo_path} -- moved 2")
                            except:
                                try:
                                    os.system(f"rm -rfv {geo_path}")
                                    print(f"{geo_path} -- removed 3")
                                except:
                                    pass
                    time.sleep(2)
                except KeyboardInterrupt:
                    sys.exit()
                except:
                    print("Run @ process:", traceback.print_exc())

                # if n == 100:
                #     break
                n+=1
        else:
            print(f"Checking path: {geo_path} >>> {os.path.exists(geo_path)}")
            try:
                if os.path.exists(geo_path):
                    file_processor = Output(model=model)
                    file_processor.process(filepath=geo_path)
                time.sleep(1)
            except KeyboardInterrupt:
                sys.exit()
            except:
                print("Run @ process:", traceback.print_exc())
    except KeyboardInterrupt:
        sys.exit()
    except:
        print("Run @@ submission:", traceback.print_exc())


if __name__ == "__main__":
    model = LMRBiLSTMAttnCRF(
        embedding_size=embed_size,
        hidden_dim=hidden_size,
        rnn_layers=rnn_layers,
        lstm_dropout=0.3,
        device=device, key_dim=64, val_dim=64,
        num_output=64,
        num_heads=16,
        attn_dropout=0.3
    )
    run(model)








