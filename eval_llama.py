import sys
from src.dataset import MERA_dataset
from argparse import ArgumentParser
from datasets import load_dataset
from src.utils import get_logger
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from tqdm import tqdm
import time
import pandas as pd
import numpy as np


import yaml

sys.path.append('.')
device = 'cuda'


def get_arguments():
    #taking arguments from the terminal
    parser = ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/config_model.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/",
        help="Path to save experiment results",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    #config file with all of the paths to data files and arguments for the model
    with open(args.config_path) as config_file:
        config = yaml.safe_load(config_file)
    logger = get_logger("train.py", config['data']['path_to_logger'])
    
    logger.info('loading the dataset')
    df = load_dataset(config["data"]["data_path"], 
                      config["data"]["dataset_name"])
    #Temporarily loding only 20 samples, remove later for full experiment
    df['train'] = df['train'].to_pandas()[0:50]
    
    logger.info('loading the model and tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["name_or_path"], 
                                              **config["tokenizer"]["kwargs"])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        **config["model"],
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    
    data = MERA_dataset(df['train'], tokenizer, config["tokenizer"]["kwargs"])
    full_dataloader = torch.utils.data.DataLoader(data, **config['loader_args'])
    
    preds = []
    start = time.time()
    logger.info('starting evaluation')
    
    with torch.no_grad():
        for batch in tqdm(full_dataloader):
            outputs = model.generate(**batch, max_new_tokens=config['generation_args']['max_new_tokens'])
            outputs = tokenizer.batch_decode(outputs)
            outputs = [i[-1] for i in outputs]
            preds.extend(outputs)
    end=time.time()    
    logger.info('finished evaluation')
    logger.info(f"inference time is: {end - start} seconds")
    logger.info(f"the accuracy is: {(np.array(preds) == df['train']['outputs']).sum()/len(preds)}")
    print('DONE')