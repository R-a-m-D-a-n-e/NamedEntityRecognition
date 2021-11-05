from transformers import AutoTokenizer,AutoConfig, TFAutoModelForTokenClassification as tfc
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from glob import glob
import numpy as np
from tqdm import tqdm
from enum import Enum
from time import time
import pickle as pkl
import re
import os
import codecs


np.random.seed(0)

class Dtype(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class NERDataset():
  def __init__(self,data_dir,dtype=Dtype.train,split_=False,encoding='utf-8',__union=False):
    self.split_ = split_
    self.encoding = encoding
    if(not __union):
      if(not type(dtype) == Dtype):
        raise ValueError("dtype doit être de type Dtype")
      self.files = glob(data_dir+"/*."+dtype.value)
      if(self.files == []):
        raise ValueError("le dossier et vide. Rq: l'extension d'un fichier *."+dtype.value)
    self.tags = []
    self.data = None
    self.text = []
  
  def union(dataset1,dataset2):
    res = NERDataset(None,None,dataset1.split_,dataset1.encoding,True)
    res.files = dataset2.files + dataset1.files
    res.tags = dataset2.tags + dataset1.tags
    res.data = dataset2.data + dataset1.data
    res.text = dataset2.text + dataset1.text

    return res

  def get_data(self):
    return self.data

  def get_text(self):
    return self.text

  def get_tags(self):
    return self.tags

  def get_unique_tags(self):
    return set([tag for tags in self.tags for tag in tags])

  def load_data(self,maxLine=None):
    data = []
    for filename in tqdm(self.files):
      # lines = [line.split() for line in file]
      # "ISO-8859-1"
      outputFile = codecs.open(filename, "r", self.encoding)
      lines = [list(np.array(line.split())[[0,-1]]) if(not re.match(r'\s+',line)) else [] for line in outputFile]
      start = 0
      for end, parts in enumerate(lines):
        str0 = []
        tags0 = []
        if not parts:
          if(self.split_):
            sample = [
                      str0.append(token) or tags0.append(tag.split('-')[-1]) or (token, tag.split('-')[-1]) 
                      for token, tag in lines[start:end]
                      ]
          else:
            sample = [
                      str0.append(token) or tags0.append(tag) or (token, tag) 
                      for token, tag in lines[start:end]
                      ]
          data.append(sample)
          self.text.append(str0)
          self.tags.append(tags0)
          start = end + 1
        if maxLine and maxLine == len(data):
          self.data = data
          return data
      # if start < end:
      #   data.append(lines[start:end])
    self.data = data
    return data


class GNERDataset():
  def __init__(self,dir_train=None,dir_dev=None,dir_test=None,maxLine=None,split_=False,encoding='utf-8'):
    self.maxLine = maxLine
    self.train = NERDataset(dir_train,Dtype.train,split_=split_,encoding=encoding)
    self.dev = NERDataset(dir_dev,Dtype.dev,split_=split_,encoding=encoding) if(dir_dev) else  dir_dev
    self.test = NERDataset(dir_test,Dtype.test,split_=split_,encoding=encoding) if(dir_test) else  dir_test
    self.load_data()

  def get_unique_tags(self):
    res = self.train.get_unique_tags()
    if(self.dev):
      res = res.union(self.dev.get_unique_tags())
    if(self.test):
      res = res.union(self.test.get_unique_tags())
    return list(res)

  def load_data(self):
    self.train.load_data(maxLine=self.maxLine)
    if(self.dev):
      self.dev.load_data(maxLine=self.maxLine)
    if(self.test):
      self.test.load_data(maxLine=self.maxLine)
    
  def get_train(self):
    return self.train.get_data() if(self.train) else None
  
  def get_text_train(self):
    return self.train.get_text() if(self.train) else None

  def get_dev(self):
    return self.dev.get_data() if(self.dev) else None
  
  def get_text_dev(self):
    return self.dev.get_text() if(self.dev) else None

  def get_test(self):
    return self.test.get_data() if(self.test) else None
  
  def get_text_test(self):
    return self.test.get_text() if(self.test) else None


class NERTokenized():
  def __init__(self,model_name_or_path,tags,seq_length=180):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tags = set(tags)
    self.seq_length = seq_length
    tags.add('_')
    self.tags = sorted(list(tags),reverse=True)
    self.tags_index = {tag: i for i, tag in enumerate(self.tags)}

  def get_tokenizer(self):
    return self.tokenizer

  def get_tags(self):
    return self.tags

  def get_itag(self,t):
    return self.tags_index.get(t,0) # tags_index[0] = '_'

  def get_tag(self,i):
    return self.tags[i] if(len(self.tags)>i) else '_'
  
  def tokenize_str_sample(self,samples):
    X = []
    for i,sample in enumerate(samples):
      tmp = self.tokenizer(sample,is_split_into_words=True)['input_ids'][1:-1][:self.seq_length-2]
      X.append([3] + tmp + [4] + [0] * (self.seq_length - len(tmp)-2))
    return  X 

  def tokenize_sample(self,sample):
    seq = [
            (subtoken, tag) for token, tag in sample 
            for subtoken in self.tokenizer(token)['input_ids'][1:-1]
          ][:self.seq_length-2]
    return [(3, 'O')] + seq + [(4, 'O')]

  def tokenized(self,data):
      tokenized_data = list(tqdm(map(self.tokenize_sample, data)))
      # max_len = max(map(len, tokenized_data))
      X = np.zeros((len(data), self.seq_length), dtype=np.int32)
      y = np.zeros((len(data), self.seq_length), dtype=np.int32)
      for i, sentence in enumerate(tokenized_data):
          for j, (subtoken_id, tag) in enumerate(sentence):
              X[i, j] = subtoken_id
              y[i,j] = self.get_itag(tag)
      return X, y


class NERModel():
  fun_aggregate = {
    'sum':np.sum,
    'max':np.max,
    'mean':np.mean
  }
  def __init__(
              self,
              model_name_or_path,
              tags,
              seq_length=180,
              num_epochs=3,
              batch_size=32,
              bert_lr=4e-5,
              epsilon=1e-08,
              agg_type='sum',
              output_dir=''
            ):
    print(tags,seq_length)
    self.metrics = None
    if(os.path.isdir(model_name_or_path)):
      metrics = [metrics for metrics in os.listdir(model_name_or_path) if  re.match(r'metrics',metrics) ]
      if(metrics):
        try:
          self.load_metrics(model_name_or_path+'/'+metrics[0])
        except:
          pass
    # create tokenizer
    self.tokenized = NERTokenized(model_name_or_path,tags,seq_length)
    # create model
    self.createModel(model_name_or_path,len(self.tokenized.get_tags()),bert_lr,epsilon)
    # props
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.agg_type = agg_type
    self.output_dir = output_dir
  
  def createModel(self,model_name_or_path,num_labels,bert_lr,epsilon):
    # create model
    config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
    self.model = tfc.from_pretrained(model_name_or_path,config=config)
    # Ajouter une couche softmax
    self.model.layers[-1].activation = tf.keras.activations.softmax
    self.model.summary()

    # compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=bert_lr,
                                              epsilon=epsilon,
                                              clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    self.model.compile(optimizer=optimizer, loss=loss, metrics='accuracy')


  def preprocessing(self,dataset):
    # tokenized
    ## train
    X_train, y_train = self.tokenized.tokenized(dataset.get_train())
    ## dev
    data_dev = None
    if(dataset.get_dev()):
      data_dev = self.tokenized.tokenized(dataset.get_dev())
      data_dev = (tf.constant(data_dev[0]),tf.constant(data_dev[1]))

    return X_train, y_train, data_dev

  def fit(self,dataset):
    X_train, y_train, data_dev = self.preprocessing(dataset)
    history = self.model.fit(tf.constant(X_train), tf.constant(y_train),
                        validation_data=data_dev,
                        validation_split=0.2 if not data_dev else None,
                        epochs=self.num_epochs, 
                        batch_size=self.batch_size)
    if(self.metrics):
      self.metrics['accuracy'] += history.history['accuracy']
      self.metrics['loss'] += history.history['loss']
      self.metrics['val_accuracy'] += history.history['val_accuracy']
      self.metrics['val_loss'] += history.history['val_loss']
    else:
      self.metrics = history.history

  def aggregate_eval(self,sample, predictions):
    results = []
    i = 1
    for token, y_true in sample:
        nr_subtoken = len(self.tokenized.get_tokenizer()(token)['input_ids']) - 2
        pred = predictions[i:i+nr_subtoken]
        i += nr_subtoken
        agg = np.argmax(NERModel.fun_aggregate.get(self.agg_type,np.sum)(pred, axis=0))
        y_pred = self.tokenized.get_tag(agg)
        results.append([token, y_true, y_pred])
    return np.array(results)

  def evaluate(self,dataset,batch_size=128):
    #dataset doit être de type NERDataset
    data_test = self.tokenized.tokenized(dataset.get_data())
    data_test = (tf.constant(data_test[0]),tf.constant(data_test[1]))

    y_probs = self.model.predict(data_test[0])[0]
    tmp = np.array([ self.aggregate_eval(sample, predictions)
                    for sample, predictions in zip(dataset.get_data(), y_probs)
                    ],dtype=object)
    y_true = []
    y_pred = []
    [ y_true.append(list(line[:,1])) or y_pred.append(list(line[:,2])) for line in tmp ]

    return y_true, y_pred

    

  def aggregate_pred(self,sample, predictions):
    results = []
    i = 1
    for token in sample:
        nr_subtoken = len(self.tokenized.get_tokenizer()(token)['input_ids']) - 2
        pred = predictions[i:i+nr_subtoken]
        i += nr_subtoken
        agg = np.argmax(NERModel.fun_aggregate.get(self.agg_type,np.sum)(pred, axis=0))
        y_pred = self.tokenized.get_tag(agg)
        results.append((token, y_pred))
    return results

  def predict_proba(self,samples,split=False):
    if(not split):
      samples = [sample.split() for sample in samples]
    X_val = self.tokenized.tokenize_str_sample(samples)

    return self.model.predict(X_val)[0]

  def predict(self,samples,split=False):
    if(not split):
      samples = [sample.split() for sample in samples]

    y_probs = self.predict_proba(samples,split=True)
    predictions = [self.aggregate_pred(sample, predictions)
               for sample, predictions in zip(samples, y_probs)]

    return predictions

  def get_metrics(self):
    return self.metrics
  
  def get_tags(self):
    return [ lab for lab in self.tokenized.get_tags() if not lab =='_']

  def save_pretrained(self,name='model_'+str(int(time()))+'.bin'):
    name_ = self.output_dir+'/'+name
    self.model.save_pretrained(name_)
    self.tokenized.get_tokenizer().save_pretrained(name_)
    self.save_metrics(name_+'/metrics.pkl')

  def save_metrics(self,name='metrics_'+str(int(time()))+'.pkl'):
    pkl.dump({'metrics':self.metrics},open(self.output_dir+name,'wb'))

  def load_metrics(self,name):
    with open(name,'rb') as f:
      self.metrics = pkl.load(f)['metrics']
      return self.metrics
  

class LPDataset(GNERDataset):
  def __init__(self,lang,dir_train,dir_dev=None,dir_test=None,perce=.8,split_=False,encoding='utf-8'):
    super().__init__(dir_train,dir_dev,dir_test,None,split_,encoding)
    self.perce = perce
    self.lang = lang

  def get_lang(self):
    return self.lang

  def set_perce(self,val):
    self.perce = val

  def get_train(self):
    data = super().get_train()
    np.random.shuffle(data)
    return data[:int(len(data)*self.perce)]
    
  def get_dev(self):
    data = super().get_dev()
    np.random.shuffle(data)
    return data[:int(len(data)*self.perce)]
    
  def get_test(self):
    data = super().get_test()
    np.random.shuffle(data)
    return data[:int(len(data)*self.perce)]

class LIDataset(GNERDataset):
  def __init__(self,lang,dir_train,dir_dev=None,dir_test=None,begin=0,end=.8,split_=False,encoding='utf-8'):
    super().__init__(dir_train,dir_dev,dir_test,None,split_,encoding)
    self.begin = begin
    self.end = end
    self.lang = lang

  def get_lang(self):
    return self.lang

  def set_interval(self,begin,end):
    self.begin = begin
    self.end = end

  def get_train(self):
    data = super().get_train()
    return data[int(len(data)*self.begin):int(len(data)*self.end)]
    
  def get_dev(self):
    data = super().get_dev()
    return data[int(len(data)*self.begin):int(len(data)*self.end)]
    
  def get_test(self):
    data = super().get_test()
    return data[int(len(data)*self.begin):int(len(data)*self.end)]

class MLDataset():
  def __init__(self,list_dataset):
    self.dataset = {data.get_lang():data for data in list_dataset}
    
  def get_unique_tags(self):
    return list(np.unique(np.concatenate([data.get_unique_tags() for data in self.dataset.values()])))

  def get_train(self):
    train = []
    for data in self.dataset.values():
      train += data.get_train()
    return train

  def get_dev(self):
    dev = []
    for data in self.dataset.values():
      dev += data.get_dev()
    return dev

  def get_test(self):
    test = []
    for data in self.dataset.values():
      test += data.get_test()
    return test

  def get_dataset_test(self):
    data = list(self.dataset.values())
    test = data[0].test
    for i in range(1,len(data)):
      test = NERDataset.union(test,data[i].test)
    return test

class MLPDataset(MLDataset):
  def set_perce(self,key,val):
    self.dataset[key].set_perce(val)

  def set_perce_all(self,val):
    for key in self.dataset.keys():
      self.dataset[key].set_perce(val)

class MLIDataset(MLDataset):
  def set_interval(self,key,begin,end):
    self.dataset[key].set_interval(begin,end)

  def set_interval_all(self,begin,end):
    for key in self.dataset.keys():
      self.dataset[key].set_interval(begin,end)

class NERModelTPU(NERModel):
  def __init__(
              self,
              model_name_or_path,
              tags,
              seq_length=180,
              num_epochs=3,
              batch_size=32,
              bert_lr=4e-5,
              epsilon=1e-08,
              agg_type='sum',
              output_dir=''
            ):
    super().__init__(
              model_name_or_path,
              tags,
              seq_length,
              num_epochs,
              batch_size,
              bert_lr,
              epsilon,
              agg_type,
              output_dir)

  def createModel(self,model_name_or_path,num_labels,bert_lr,epsilon):
    try:
      device_name = os.environ['COLAB_TPU_ADDR']
      TPU_ADDRESS = 'grpc://' + device_name
      print('Found TPU at: {}'.format(TPU_ADDRESS))
    except KeyError as e:
      print('TPU not found')
      raise e
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)
    with strategy.scope():
      super().createModel(model_name_or_path,num_labels,bert_lr,epsilon)

