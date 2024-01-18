import torch
import tensorflow
from torch import nn
import time
from torchmetrics.classification import BinaryAccuracy



from datasets import load_dataset
from transformers import pipeline

token='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'






print(torch.cuda.is_available())
print(torch.cuda.device_count())


####################################
########### LOAD DATASET ###########
####################################



############################### COLA ###############################
cola_train= load_dataset('glue', 'cola', split="train")
cola_val= load_dataset('glue', 'cola', split="validation")
cola_test= load_dataset('glue', 'cola', split="test")

#print(cola_val)
#print(cola_test)


############################### STSB  ###############################

'''
stsb_multi_mt_train = load_dataset("stsb_multi_mt", name="en", split="train")
stsb_multi_mt_dev = load_dataset("stsb_multi_mt", name="en", split="dev")
stsb_multi_mt_test = load_dataset("stsb_multi_mt", name="en", split="test")

'''
####################################
###########  TOKENIZER   ###########
####################################
from transformers import BertTokenizer, DataCollatorWithPadding


tokenizer = BertTokenizer.from_pretrained('Abirate/bert_fine_tuned_cola', truncation=True, padding=True,)


def tokenize_function(example):
    return tokenizer(cola_val['sentence'], truncation=True, padding=True, return_tensors="pt" )


tokenized_dataset = cola_val.map(tokenize_function, batched=True, batch_size=1043)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer,  padding=True,)
tokenized_dataset = tokenized_dataset.remove_columns(["sentence",  "idx"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")

#print(tokenized_dataset)



####################################
###########  DATALOADER  ###########
####################################
from torch.utils.data import DataLoader



eval_dataloader = DataLoader(
    tokenized_dataset, batch_size=8, collate_fn=data_collator)




# for batch in eval_dataloader:
#   print({k: v.shape for k, v in batch.items()})


################################################
###########       GPU AND CUDA       ###########
################################################

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)





################################################
########### MODEL ###########
################################################

from transformers import BertForSequenceClassification
Softmax = nn.Softmax()


model = BertForSequenceClassification.from_pretrained('Abirate/bert_fine_tuned_cola',token= token,from_tf=True, num_labels=2)
#model.to(device)



from tqdm.auto import tqdm
progress_bar = tqdm(range(len(eval_dataloader)))


import evaluate

metric = evaluate.load("glue", "cola")

predictions=torch.empty(0)
for batch in eval_dataloader:
  batch = {k: v for k, v in batch.items()}
  #batch = {k: v.to(device) for k, v in batch.items()}
  with torch.no_grad():
    outputs = model(**batch)
  #outputs= model(**batch) ######  time this and with torch no_grade since you are using it only for inference you dont need gradient
  #print(outputs,'\n')
  #outputs1= Softmax(outputs.logits)
  #print(outputs1,'\n')
  outputs2= (torch.argmax(Softmax(outputs.logits), dim=-1))
  #print(outputs2.size())
  predictions=torch.cat((predictions, outputs2),dim=0)
  #print(outputs2,'\n')
  #print(batch["labels"])
  progress_bar.update(1)
  metric.add_batch(predictions=outputs2, references=batch["labels"])
result= metric.compute()
print(result)
print(predictions[0:25],'\n')
print(tokenized_dataset["labels"][0:25],'\n')
print(tokenized_dataset["labels"][0:25]-predictions[0:25],'\n')

metric2 = BinaryAccuracy()
print(metric2(predictions, tokenized_dataset["labels"] ))



'''
print(model.device)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
#tokenizer.to(device)
#cola_val.to(device)


start = time.time()
Bert_predictions=[]
True_Lables=[]
for i in range(len(cola_val)):
  input= tokenizer(cola_val[i]['sentence'], padding=True, return_tensors="pt" )
  
  output = model(**input)
  Bert_predictions.append(Softmax(output.logits[0]))
  True_Lables.append(cola_val[i]['label'])
  #DistilBert_similarity_score.append(round(stsb_multi_mt_test[i]['similarity_score'],2))
  #print(cola_val[i])
  #print(Bert_predidddctions)

end = time.time()
print(end - start)

#print(Bert_predictions)
#print(True_Lables)

'''









'''
from transformers import    DistilBertForSequenceClassification,   DistilBertTokenizer
model =  DistilBertForSequenceClassification.from_pretrained('vicl/distilbert-base-uncased-finetuned-stsb',token= token,num_labels=1)
tokenizer = DistilBertTokenizer.from_pretrained('vicl/distilbert-base-uncased-finetuned-stsb',token= token)

print(model)
print(tokenizer)




DistilBert_predictions=[]
DistilBert_similarity_score=[]
for i in range(100):
  input= tokenizer(stsb_multi_mt_test[i]['sentence1'], stsb_multi_mt_test[i]['sentence2'], padding=True,return_tensors
  output = model(**input)="pt" )
  DistilBert_predictions.append(round(output.logits[0].item(),2))
  DistilBert_similarity_score.append(round(stsb_multi_mt_test[i]['similarity_score'],2))

print(DistilBert_predictions)
print(DistilBert_similarity_score)

'''