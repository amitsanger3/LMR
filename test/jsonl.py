import os, json
import ast
from utils import lmr_logger
import traceback


class Jsonl(object):

    def __init__(self, filepath="/geoai/input.jsonl"):
        self.input_filepath = filepath
        self.output_filepath =  filepath.replace('input', 'output')

    def read(self):
        result = []
        true_records = 0
        false_records = 0
        try:
            with open(self.input_filepath, 'r') as op:
                for json_obj in list(op):
                    try:
                        result.append(json.loads(json_obj.strip()))
                        true_records += 1
                    except:
                        lmr_logger.log_obj.error(traceback.print_exc(), exc_info=True)
                        try:
                            result.append(ast.literal_eval(json_obj.strip()))
                            true_records += 1
                        except:
                            lmr_logger.log_obj.error(traceback.print_exc(), exc_info=True)
                            false_records += 1
                            continue
            op.close()
        except:
            lmr_logger.log_obj.error(msg="Error in reading file.", exc_info=True)
        print(
            f" Total records:{true_records + false_records}\n System read successfully:{true_records}\n Failed records:{false_records}")
        return result

    def write(self, output_list):
        true_records = 0
        false_records = 0
        try:
            with open(self.output_filepath, 'w+') as op:
                for json_obj in output_list:
                    try:
                        json.dump(json_obj, op)
                        true_records += 1
                        op.write('\n')
                    except:
                        lmr_logger.log_obj.error(traceback.print_exc(), exc_info=True)
                        false_records += 1
            op.close()
        except:
            lmr_logger.log_obj.error(traceback.print_exc(), exc_info=True)
        print(
            f" Total records:{true_records + false_records}\n System write successfully:{true_records}\n Failed records:{false_records}")
        return None