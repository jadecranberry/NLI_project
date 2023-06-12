import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from typing import List
from tqdm import tqdm
import os
import pickle as pkl
import tensorflow_hub as hub

class EmbeddingGenerator:
    def __init__(self, model_name: str,
                 device: torch.device, tf: bool):
        self.device = device
        self.model = hub.load("https://tfhub.dev/google/" + model_name) if tf else SentenceTransformer(model_name) 
        self.model_path = model_name
        if 'model/' not in model_name:
            self.model_path = 'model/' + model_name
        self.tf = tf

    def __call__(self, documents: List[dict]):
        print("Generating embeddings.")
        i=0
        for document in tqdm(documents):
             print(document['url'])
             if self.tf:
                sent_embeddings = torch.from_numpy(self.model(document['abstract']).numpy())
             else:
                with torch.no_grad():
                   sent_embeddings = self.model.encode(document['abstract'],num_workers=1,batch_size=100,show_progress_bar=True,convert_to_numpy=False)
                
             document = document.copy()
             document['embeddings'] = sent_embeddings
                
             if not os.path.isdir('./embeddings/' + self.model_path):
                  os.makedirs('./embeddings/' + self.model_path)

             file_name = './embeddings/'+ self.model_path + '/document' + str(i) + '.pkl'
             with open(file_name,'wb') as f:
                   pkl.dump(document,f)
             i+=1
                
        return 0
