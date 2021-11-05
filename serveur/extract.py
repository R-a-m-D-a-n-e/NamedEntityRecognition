import numpy as np
import json
import re
import src.NERModelServeur as ner

tags = ['O', 'B-PER', 'B-MISC', 'I-ORG', 'B-LOC', 'I-LOC', 'I-PER', 'B-ORG', 'I-MISC']
NAME_MODEL = 'model/model80_90.bin'
model_esp = ner.NERModel(NAME_MODEL,tags,num_epochs=10)

# models = {
#     "Allemand":None
#     , "Anglais":None
#     , "Espagnol":model_esp
#     , "Français":None
#     , "Nerlande":None
# }

def getData(requestJson):
    opt = requestJson.get('opt')
    query = requestJson.get('text')
    return opt, query


def extractEntity(opt,text):
    # model = models.get(opt,model_esp)
    model = model_esp
    if(len(text)<1):
        return json.loads(json.dumps({'entitys':[],'state': 'ko'}))
    res = model.predict(text)
    res = [{'value':entity[0],'tag':entity[1]} for entity in res[0] if entity[1]!='O']
    return json.loads(json.dumps({'entitys':res,'state': 'ok'}))

