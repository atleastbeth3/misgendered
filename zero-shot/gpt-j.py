import argparse
from pathlib import Path
import pandas as pd
import torch
import re
import os
import sys
sys.path.append('/extra/ucinlp0/tthossai/package')

from transformers import AutoTokenizer, AutoModelForCausalLM

gender_codebook=pd.read_csv('pronouns.csv')
gender_codebook.columns=['gender_type', 'gender', 'nom', 'acc', 'pos_dep', 'pos_ind', 'ref']

#Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--eval_file', type=Path) 
parser.add_argument('--output_path', type=Path) 
    
args = parser.parse_args()
eval_file=args.eval_file
output_path=args.output_path 

#Load
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B", cache_dir='cache/')
#model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", torch_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", cache_dir='cache/')

model.eval()
model.cuda()

def declare_pronouns(gender, sentence):
  row=gender_codebook[gender_codebook['gender']==gender]
  row=row.reset_index()
  sentence=re.sub('{nom}', row['nom'][0],sentence)
  sentence=re.sub('{acc}', row['acc'][0],sentence)
  sentence=re.sub('{pos_dep}', row['pos_dep'][0],sentence)
  sentence=re.sub('{pos_ind}', row['pos_ind'][0], sentence)
  sentence=re.sub('{ref}', row['ref'][0],sentence)
  return sentence

def get_label(gender, form):
  row=gender_codebook[gender_codebook['gender']==gender]
  row=row.reset_index()
  label=row[form][0]
  return label

def label_to_cap(sentence):
  mask_start=sentence.find('{')
  if sentence[mask_start-2]=='.':
    return True
  else:
    False

def get_eval_instance(sent_piece, gender, form):
  sentence=declare_pronouns(gender, sent_piece)
  label=get_label(gender, form)
  constraints=gender_codebook[form].tolist()

  if label_to_cap(sentence):
      label=label.capitalize()
      constraints=[c.capitalize() for c in constraints]
  return sentence, label, constraints

def evaluate(eval_path, gender):
  eval=pd.read_csv(eval_path)

  eval['gender']=''
  eval['label']=''
  eval['top_1']=''
  eval['top_2']=''
  eval['top_3']=''
  eval['loss_1']=0
  eval['loss_2']=0
  eval['loss_3']=0
  eval['correct']=0
  
  #For each instance
  for idx, instance in eval.iterrows():
    print (idx)
    #Get eval instance
    sent_piece=instance['sentence']
    form=instance['form']
    sentence, label, constraints=get_eval_instance(sent_piece, gender, form)
    eval.at[idx, 'sentence']=sentence
    eval.at[idx, 'label']=label

    #Evaluate
    losses=[]
    for c in constraints:
        choice=re.sub('{mask_token}', c, sentence)
        input_ids=tokenizer(choice, return_tensors="pt")["input_ids"].to('cuda')
        out=model(input_ids, labels=input_ids)  
        loss=out['loss'].detach().item()
        losses.append(loss)
      
    words=pd.DataFrame({'labels': constraints, 'losses':losses})
    words = words.sort_values('losses', ascending=True, ignore_index=True)

    eval.at[idx, 'top_1']=words.labels[0]
    eval.at[idx, 'top_2']=words.labels[1]
    eval.at[idx, 'top_3']=words.labels[2]

    eval.at[idx, 'loss_1']=words.losses[0]
    eval.at[idx, 'loss_2']=words.losses[1]
    eval.at[idx, 'loss_3']=words.losses[2]

    if words.labels[0]==label:
      eval.at[idx, 'correct']=1

  return eval

def main():
        
    #Evaluate
    for gdx, row in gender_codebook.iterrows():
      gender=row['gender']
      file_name=gender+'.csv'
      file_path=os.path.join(output_path,file_name)
      if not os.path.exists(file_path):

      	print(gender)
      	eval=evaluate(eval_file, gender)
    
      	#Save Raw results
      	eval.to_csv(file_path)
    
if __name__ == '__main__':
    main() 