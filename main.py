from utils import load_yaml, training_gen
import pickle
import os
import tensorflow as tf
import numpy as np
import random as rn
from tensorflow.keras import layers

def define_model(n_timesteps):
    inputs = layers.Input(shape=(n_timesteps,1)) 
    lstm_out1 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inputs)
    lstm_out2 = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(lstm_out1)    
    outputs = layers.Dense(1, activation = 'sigmoid')(lstm_out2)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def run(yaml_file):      
    
    args = load_yaml(yaml_file)   

    print('data' + str(args.data))     
        
    n_timesteps = int(args.preprocess.resample_fs*args.data.sample_time)   
                
    generator = training_gen(args)      
    
    val_x = 'val_X_smoke.npy'
    val_y = 'val_Y_smoke.npy'    
    print(f'Loading validation: {val_x, val_y}')      
    X_val = np.load(os.path.join(os.getcwd(),'io','realdata', val_x))
    Y_val = np.load(os.path.join(os.getcwd(),'io','realdata', val_y))  
    
    tf.keras.backend.clear_session()
    model = define_model(n_timesteps)   
    opt = tf.keras.optimizers.Adam(learning_rate=0.0003)   
    model.compile(loss='binary_crossentropy', optimizer = opt,\
                  metrics=[tf.keras.metrics.AUC(curve = 'ROC'), 
                           tf.keras.metrics.AUC(curve = 'PR'),
                           tf.keras.metrics.Recall(), 
                           tf.keras.metrics.Precision()])
    model.summary()

    history = model.fit(generator, steps_per_epoch=args.train.steps,
              epochs=args.train.epochs,
              validation_data = (X_val, Y_val))   

    filename = os.path.basename(yaml_file)
    name = os.path.splitext(filename)[0]
       
    history_file = os.path.join(os.getcwd(),'io','histories', name + '.pickle')
    with open(history_file, 'wb') as handle:
        pickle.dump(history.history, handle, protocol=pickle.HIGHEST_PROTOCOL)        
        
    if args.train.savemodel:
        model.save(os.path.join(os.getcwd(),'io','neuralnet_models', name +'.h5'))
        print(f'Model saved as: {name}'+'.h5')                


if __name__ == "__main__":       

    import gc
    import sys

    def run_one(filename):    
        np.random.seed(1234)
        rn.seed(1234)
        tf.random.set_seed(1234)   
        gc.collect()
        run(os.path.join(os.getcwd(),'io', 'config_files',filename))      

    run_one(sys.argv[1])



