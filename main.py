import pandas as pd
from model import DLModels
from trainer import CustomTrainer
import tensorflow as tf
from common_funcs import sliding_window, ColorPrint, add_features, collect_new_open_data
import yaml
import argparse
from datetime import datetime
import os


def import_data(symbol:str):
    if os.path.exists(f"trainingDS/{symbol}/alldata.csv") == False:
        data = collect_new_open_data(symbol = symbol, start_date=datetime(2023,12,8))
    else:
        data = pd.read_csv(f"trainingDS/{symbol}/alldata.csv", parse_dates = ['timestamp'])
    data = add_features(data)
    
    return data
    
def PreTrain_Model(model, symbol:str):
    data = import_data(symbol)
    if len(data) > 300000:  # Breaks dataset into chunks if too large to fit into memory
        start, end = 0.0, 0.10
        while start < 1.0:
            sub_data = data[int(len(data) * start):int(len(data) * end)]
            if len(sub_data) <= 1: break
            print(ColorPrint(f"PRETRAIN {symbol} ({int(10*start)+1}/10)", "Yellow"))
            print(f'=' * 100)
            X,y,_ = sliding_window(sub_data, keys['attributes']['sequence_len'])
            trainer = CustomTrainer(model, X, y, pretrain_parameters['output_file'])
            print(ColorPrint("Loading: ", "Blue"), ColorPrint(f"{keys['model'].upper()} Model", "Green"))
            trainer.train_model(num_epochs=keys['num_epochs'], batch_size=pretrain_parameters['batch_size'])
            start = end
            end += 0.10
            print(f'=' * 100)
    else:
        print(ColorPrint(f"PRETRAIN {symbol}", "Yellow"))
        print(f'=' * 100)
        X,y,_ = sliding_window(data, keys['attributes']['sequence_len'])
        trainer = CustomTrainer(model, X, y, pretrain_parameters['output_file'])
        print(ColorPrint("Loading: ", "Blue"), ColorPrint(f"{keys['model'].upper()} Model", "Green"))
        trainer.train_model(num_epochs=keys['num_epochs'], batch_size=pretrain_parameters['batch_size'])
        print(f'=' * 100) 

def FineTune_Model(symbol):
    model = tf.keras.models.load_model(f'{pretrain_parameters["output_file"]}', compile = False)
    print(f'=' * 100)
    data = import_data(symbol)
    if len(data) > 300000:
        start, end = 0.0, 0.05
        while start < 1.0:
            sub_data = data[int(len(data) * start):int(len(data) * end)]
            if len(sub_data) <= 1: break
            print(ColorPrint(f"FINE-TUNE {symbol} ({int(20*start)+1}/20)", "Yellow"))
            print(f'=' * 100)
            X,y,_ = sliding_window(sub_data, keys['attributes']['sequence_len'])
            print(ColorPrint("Loading: ", "Blue"), ColorPrint(f"{keys['model'].upper()} Model", "Green"))
            trainer = CustomTrainer(model, X, y, fine_tune_parameters['output_file'])
            trainer.train_model(num_epochs=keys['num_epochs'], batch_size=fine_tune_parameters['batch_size'])  
            start = end
            end += 0.05
            print(f'=' * 100)
    else:
        print(ColorPrint(f"FINE-TUNE", "Yellow"))
        print(f'=' * 100)
        X,y,_ = sliding_window(data, keys['attributes']['sequence_len'])
        print(ColorPrint("Loading: ", "Blue"), ColorPrint(f"{keys['model'].upper()} Model", "Green"))
        trainer = CustomTrainer(model, X, y, fine_tune_parameters['output_file'])
        trainer.train_model(num_epochs=keys['num_epochs'], batch_size=fine_tune_parameters['batch_size'])
        print(f'=' * 100) 
        

        
if __name__ == '__main__':
    config_path = 'configs'
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Enter Config File Name')
    args = parser.parse_args()
    cfg = args.cfg

    with open(os.path.join(config_path, cfg), 'r') as file:
        keys = yaml.safe_load(file)

    base_parameters = keys.get('attributes', {})
    pretrain_parameters = keys.get('pretrain',{})
    fine_tune_parameters = keys.get('fine_tune',{})
    pretrain_parameters = {**base_parameters, **pretrain_parameters}
    fine_tune_parameters = {**base_parameters, **fine_tune_parameters}

    for paramType, param in {'PRETRAIN': pretrain_parameters, 'FINE-TUNE': fine_tune_parameters}.items():
        print(ColorPrint(f'\n{paramType} PARAMETERS', 'Yellow'))
        for key, value in param.items():
            print(ColorPrint(f'   {key}: ', 'Cyan'), value)


    model_type = keys['model'].lower()
    symbols = keys['pretrain']['dataset']

    if model_type == 'model_1':
        model = DLModels(**pretrain_parameters).model_1()
    else:
        print(ColorPrint('Invalid Model Type.  Retry with options: [model_1]', 'Red'))
        exit()


    for symbol in symbols:
        PreTrain_Model(model, symbol)
    FineTune_Model(keys['fine_tune']['dataset'])