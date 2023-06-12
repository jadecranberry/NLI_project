import argparse
import jsonlines

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, required=True)
parser.add_argument('--abstract-retrieval', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
abstract_retrieval = jsonlines.open(args.abstract_retrieval)
output = jsonlines.open(args.output, 'w')

for doc in abstract_retrieval:
    evidence = {}

    for doc_id in doc['doc_ids']:
        evidence[doc_id] = [len(corpus[doc_id]['abstract']) - 1]

    output.write({
        'claim_id': doc['claim_id'],
        'evidence': evidence
    })
