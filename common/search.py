import time
import csv
import numpy as np

from common.utils import make_dataframe
# from tensorflow.keras.optimizers import SGD

class ConfigsTester:
    def __init__(self, spacepoints_number_space, hiddenlayers_number_space, hiddenlayers_neurons_number_space, hiddenlayers_activation_func_space, outputlayer_activation_func_space, metric = 'loss', tail_size = 20, save_path='/host/dissertation/trained_models/'):
        
        import itertools
        self.config_space = list(itertools.product(spacepoints_number_space,hiddenlayers_number_space,hiddenlayers_neurons_number_space,hiddenlayers_activation_func_space, outputlayer_activation_func_space))
        self.param = {'spacepoints_number': 0, 'hidden_layers': 1, 'neurons_num': 2, 'hidden_activation_func': 3, 'output_activation_func': 4}
        
        import os
        self.WORKING_DIR = save_path + f'{int(time.time())}/'
        os.umask(0)
        os.mkdir(self.WORKING_DIR, mode=0o777)
        os.mkdir(self.WORKING_DIR + 'history/', mode=0o777)
        os.mkdir(self.WORKING_DIR + 'weights/', mode=0o777)
        
        self.HP_FILEPATH = self.WORKING_DIR + 'hp.csv'
        with open(self.HP_FILEPATH, 'w', newline='', opener=lambda path, flags: os.open(path, flags, 0o777)) as f:
            hp_file_columns = ['hp_config','time',f'val_{metric}-avg',f'val_{metric}-med',f'val_{metric}-std',f'train_{metric}-max',f'train_{metric}-min']
            csv.writer(f).writerow(hp_file_columns)
            
        self.metric = metric
        self.tail_size = tail_size

    @staticmethod
    def get_config_string(config):
        return str(config).translate(str.maketrans(',','-'," '()"))
    @staticmethod
    def get_historyfile_subpath(config):
        return 'history/' + ConfigsTester.get_config_string(config) + '-history.csv'
    @staticmethod
    def get_weightsfile_subpath(config):
        return 'weights/' + ConfigsTester.get_config_string(config) + '-weights.h5'
    
    def append_hp_file(self, config, history_df):
        val_metric_tail = history_df[f'val_{self.metric}'][-self.tail_size:]
        metric_tail = history_df[self.metric][-self.tail_size:]
        
        with open(self.HP_FILEPATH, 'a', newline='') as f:
            csv.writer(f).writerow([self.get_config_string(config),time.strftime("%H:%M:%S", time.localtime()),np.average(val_metric_tail),np.median(val_metric_tail),np.std(val_metric_tail),np.max(metric_tail),np.min(metric_tail)])
    
    @staticmethod
    def load_config_json(compile_conf_dir):
        import json, ast
        with open(compile_conf_dir + 'config.json') as f:
            config = json.load(f)
            config['optimizer'] = ast.literal_eval(config['optimizer'])
            return config
    
    def save_config_json(self, optimizer, loss, epochs, scenario):
        import json, os
        with open(self.WORKING_DIR + 'config.json', 'w', newline='', opener=lambda path, flags: os.open(path, flags, 0o777)) as f:
            # config = {'optimizer': str(model.optimizer.get_config()), 'loss': model.loss, 'epochs': epoch}
            config = {'optimizer': str(optimizer.get_config()), 'loss': loss, 'epochs': epochs, 'scenario': scenario}
            f.write(json.dumps(config))
    
    def test_configs(self, y_data, optimizer_factory, loss, epochs, scenario = None, callbacks=[], verbose=True, x_data_dir = '/host/dissertation/proccessed_data/'):
        
        self.save_config_json(optimizer_factory(), loss, epochs, scenario)
        
        import cvnn.layers as complex_layers
        from tensorflow.keras.models import Sequential
        for config in self.config_space:
            print('START', config)
            
            model = Sequential()
            model.add(complex_layers.ComplexInput(input_shape=(config[self.param['spacepoints_number']],)))
            for layer_no in range(config[self.param['hidden_layers']]):
                model.add(complex_layers.ComplexDense(units=config[self.param['neurons_num']], activation=config[self.param['hidden_activation_func']]))
            model.add(complex_layers.ComplexDense(units=np.shape(y_data)[-1], activation=config[self.param['output_activation_func']]))
            if self.metric != 'loss':
                model.compile(optimizer=optimizer_factory(), loss=loss, metrics=[self.metric])
            else:
                model.compile(optimizer=optimizer_factory(), loss=loss)
                
            print('      model compiled')
            points = np.load(x_data_dir + 'points_' + str(config[self.param['spacepoints_number']]) + '.npy')
            print('      training...')
            history = model.fit(points, y_data, epochs=epochs, validation_split=0.2, verbose=0, callbacks=callbacks)
            print('      model trained')
            df = make_dataframe(history)
            self.append_hp_file(config, df)
            print('      hp_file appended')
            df.to_csv(self.WORKING_DIR + self.get_historyfile_subpath(config), index=False)
            print('      history saved')
            model.save_weights(self.WORKING_DIR + self.get_weightsfile_subpath(config), save_format='h5')
            print('      weights saved')
            
            print('DONE ', config)
