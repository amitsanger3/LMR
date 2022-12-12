import warnings
warnings.filterwarnings("ignore")
import sys
import traceback

# importing self dependencies
from config import *
from utils import *
from dataset import FlairDetection, SaveDataset
from .jsonl import Jsonl

import torch
from tqdm import tqdm


class Output(object):

    def __init__(self, model):
        self.model = self.load_model(model)
        self.trainable_params_info = True
        self.fd = FlairDetection(base_model=bert_base_model, base_model2=base_model, multi_embed=True)
        self.looger = lmr_logger.log_obj

    def print_params(self):
        try:
            pytorch_total_params = sum(p.numel() for p in self.model.parameters())
            print("Total Params:", pytorch_total_params)

            pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("Total Trainable Params:", pytorch_total_params)
        except:
            self.looger.error(traceback.print_exc(), exc_info=True)

    def load_model(self, model):
        try:
            model_name = os.path.join(model_path, "lmr.pt")

            if not os.path.exists(model_name):
                print(f"""Pretrained model required for location detection on path:{model_name}. Please train your weights first. """)
                sys.exit()
            model.load_state_dict(torch.load(model_name, map_location=device))
            model.to(device)
            return model
        except:
            self.looger.error(traceback.print_exc(), exc_info=True)

    def unit(self, input_data, model):
        try:
            blog = input_data["text"]
            blod_id = input_data["tweet_id"]
            fdr = self.fd.sentance_embeddings(blog)
            predictions = detect_fn(data=fdr, model=model, device=device, batch_size=batch_size)
            results = {
                "tweet_id": blod_id,
                "location_mentions": []
            }
            sum_pred = 0
            sum_label = 0
            for i in range(len(fdr["words"])):
                prediction_op = predictions[0][i].item()
                # print("       ", fdr["words"][i], prediction_op)
                sum_pred += prediction_op
                try:
                    word_ = fdr["words"][i]
                    word = word_["text"]
                except:
                    continue
                if prediction_op > 0.5 and word != 'PAD':
                    sum_label += prediction_op

                    word_start = word_["start_offset"]
                    word_end = word_["end_offset"]
                    if len(results["location_mentions"]) > 0:
                        prev = results["location_mentions"].pop()
                        prev_word = prev["text"]
                        prev_start = prev["start_offset"]
                        prev_end = prev["end_offset"]

                        if prev_end == word_start:
                            word = prev_word + word
                            word_start = prev_start

                        elif prev_end + 1 == word_start:
                            word = prev_word + ' ' + word
                            word_start = prev_start

                        else:
                            results["location_mentions"].append(prev)

                    results["location_mentions"].append({
                        "text": word,
                        "start_offset": word_start,
                        "end_offset": word_end
                    })
            return results
        except:
            self.looger.error(traceback.print_exc(), exc_info=True)

    def process(self, filepath="/geoai/input.jsonl"):
        '''
        Return output_data and make seperate function for
        writing file.
        :param filepath:
        :return:
        '''
        try:
            file_obj = Jsonl(filepath=filepath)
            blog_list = file_obj.read()
            output_data = []
            model = self.model
            true_records = 0
            total_records = len(blog_list)
            for blog_data in tqdm(blog_list, desc='Detecting Locations >>> ', total=len(blog_list)):
                try:
                    unit_output = self.unit(input_data=blog_data, model=model)
                    if unit_output:
                        output_data.append(unit_output)
                        true_records += 1
                    else:
                        print("This blog dumped due to above printed error.")
                        print("This blog dumped due to above printed error.")
                except:
                    self.looger.error(traceback.print_exc(), exc_info=True)
            file_obj.write(output_list=output_data)
            print(f"""Output is saved to: {file_obj.output_filepath} with LMR Detection 
            records: {true_records} out of: {total_records} given records""")
        except:
            self.looger.error(traceback.print_exc(), exc_info=True)
        return None






