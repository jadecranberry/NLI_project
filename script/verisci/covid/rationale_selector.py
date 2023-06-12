import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from typing import List
from tqdm import tqdm
import os
import pickle as pkl

import tensorflow_hub as hub


class RationaleSelector:
    def __init__(self, model: str, selection_method: str, threshold: float,
                 device: torch.device, tf: bool):
        self.device = device
        #self.tokenizer = AutoTokenizer.from_pretrained(model)
        #self.model = AutoModelForSequenceClassification.from_pretrained(model).eval().to(self.device)
        self.model_name = model

        if tf:
           self.model = hub.load("https://tfhub.dev/google/" + model)
           self.tf = tf
        else:
           if os.path.isdir("model/" + model):
              model = "model/" + model
           self.model = SentenceTransformer(model)
           self.tf = tf

        self.selection_method = selection_method
        self.threshold = threshold

    def __call__(self, claim: str, documents: List[dict], k=3,from_disk=False):
        print("Selecting rationales.")
        results = []
        if self.tf:
           claim_embedding = torch.from_numpy(self.model([claim]).numpy())
        else:
           with torch.no_grad():
              claim_embedding = self.model.encode([claim])

        if from_disk:
             for file in os.listdir("embeddings/model/" + self.model_name):
                  with open('embeddings/model/' + self.model_name + '/' + file,'rb') as f:
                      document = pkl.load(f)
                      sent_embeddings = list(document['embeddings'])
                      matches = util.semantic_search(claim_embedding, sent_embeddings, top_k=k)
                      document = document.copy()
                     # document['evidence'] = [d['corpus_id'] for d in matches[0] if d['score']>self.threshold]
                     # document['evidence_confidence'] = [d['score'] for d in matches[0] if d['score']>self.threshold]
                      document['evidence'] = [d['corpus_id'] for d in matches[0]] 
                      document['evidence_confidence'] = [d['score'] for d in matches[0]]
                      results.append(document)
        else:
             for document in tqdm(documents):
                #encoded_dict = self.tokenizer.batch_encode_plus(
                #    zip(document['abstract'], [claim] * len(document['abstract'])),
                #    pad_to_max_length=True,
                #    return_tensors='pt'
                #)
                #encoded_dict = {key: tensor.to(self.device) for key, tensor in encoded_dict.items()}
                #evidence_logits = self.model(**encoded_dict)[0]
                #evidence_scores = torch.sigmoid(evidence_logits[:, 1]).cpu().numpy()
                if self.tf:
                   sent_embeddings = torch.from_numpy(self.model(document['abstract']).numpy())
                else:
                   sent_embeddings = self.model.encode(document['abstract'],batch_size=100,show_progress_bar=True)
                   
                matches = util.semantic_search(claim_embedding, sent_embeddings, top_k=k)
                

                  #if self.selection_method == "threshold":
                  #    keep = matches if matches['score'] > self.threshold
                #    evidence_indices = sorted(keep.nonzero()[0].tolist())
                 # else:
                 #     keep = matches 

                document = document.copy()
                document['evidence'] = [d['corpus_id'] for d in matches[0]]
                document['evidence_confidence'] = [d['score'] for d in matches[0]]
                results.append(document)
        return results
