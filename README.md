# NLI_project
This repository contains code for the Postdoctoral Fellowship project at Factually Health Inc.

## Table of contents
- [Introduction](#Introduction)
- [Method](#Method)
- [Dependencies](#dependencies)
- [Download pre-trained models](#download-pre-trained-models)
- [Training scripts](#training-scripts)
- [Verify claims about COVID-19](#verify-claims-about-covid-19)
- [Citation](#citation)
- [Contact](#contact)


## Introduction

During the COVID-19 pandemic, lots of new findings were generated in the scientific literature at an unprecedented speed. It is critical to evaluate the veracity of scientific claims to avoid the problems created by making decisions based on outdated and incomplete information. However, faced with such enormous and abundant resources in scientific publications, it is impractical to evaluate all information manually. Therefore, an automatic tool for verifying this information is in demand.
 
This task of Claim verification against textual sources is referred to as verification or fact checking, in which we verify an input claim against a corpus of documents that support or refute the claim. This task is different from Textual Entailment (TE) and Natural Language Inference (NLI). For TE and NLI, the evidence is already available and in the format of a text fragment, or a single sentence in more recent years. For the later case, the claim and the evidence together are called a sentence pair. Whereas for claim verification, the evidence is not readily available and needs to be retrieved from a large set of documents.
 
Due to the fact that the global truth label to a scientific claim is hard to obtain  (cause it requires a systematic review by a team of experts), our task of scientific claim verification focuses on assigning Support or Refute to the claims instead of the true/false the claim itself is.

State of art NLP models are based on opaque neural network models which  causes interpretability (reasoning of model outputs) issues for NLP. In this paper we focus on the interpretable models in NLP. The key to interpretability of NLP models is “rationales” (supporting evidence) which are usually snippets that support outputs. We use rationales to justify each Support / Refutes decision. 

In this work, we propose our own method for scientific verifications. A three-step pipeline was propsed. In the first step, abstracts relating to an input claim are retrieved from the CORD-19 corpus. In the second step, rationale sentences are selected using semantic search. In the final step, labels are generated indicating if the abstracts support, refute or have not enough information to determine the claim. This last step is more like NLI. 

## Method

- Abstract retrieval: this component queries the covidex searching engine with the claim and retrieves the k most similar abstracts to the claim. In this process, term frequency-inverse document frequency (TF-IDF) similarity scores are calculated and abstracts with the highest scores are extracted.
- Rationale selection: this step extracts the rationale sentences (top k) by comparing the claim with each sentence of an abstract. In this paper, we will restrict the comparisons to the abstracts. Yet, owing to the low computational cost of semantic search, comparing every sentence of a document (not just the abstract) is possible and could be accomplished in a very short time (which would be our future work).
- Label prediction: this step judges whether the document supports, refutes or contains no information for the claim based on the extracted rationale sentences from the previous step. 

![image](https://github.com/jadecranberry/NLI_project/assets/32283596/9bfd6685-5cd0-40ed-a12b-fb5d493b0226)


## Dependencies

We recommend you create an anaconda environment:
```bash
conda create --name NLI_project python=3.7 conda-build
```
Then, from the `NLI_project` project root, run
```
conda develop .
```
which will add the NLI_project code to your `PYTHONPATH`.

Then, install Python requirements:
```
pip install -r requirements.txt
```

## Download pre-trained models

You can download the models using the script:
```bash
./script/download-model.sh rationale roberta_large scifact
./script/download-model.sh label roberta_large fever_scifact
```
The script checks to make sure the downloaded model doesn't already exist before starting new downloads.



## Training scripts

See [training.md](doc/training.md).

## Verify claims about COVID-19

While the project [website](https://scifact.apps.allenai.org) features a COVID-19 fact-checking demo, it is not configurable and uses a "light-weight" version of VeriSci based on [DistilBERT](https://arxiv.org/abs/1910.01108). We provide a more configurable fact-checking script that uses the full model. Like the web demo, it uses [covidex](https://covidex.ai) for document retrieval.  Usage is as follows:

```shell
python script/verify_covid.py [claim-text] [report-file] [optional-arguments].
```

For a description of the optional arguments, run `python script/verify_covid.py -h`. The script generates either a `pdf` or `markdown` report. The `pdf` version requires [pandoc](https://pandoc.org) and [wkhtmltopdf](https://wkhtmltopdf.org), both of which can be installed with `conda`. A usage example might be:

```shell
python script/verify_covid.py \
  "Coronavirus droplets can remain airborne for hours" \
  results/covid-report \
  --n_documents=100 \
  --rationale_selection_method=threshold \
  --rationale_threshold=0.2 \
  --verbose \
  --full_abstract
```

The 36 claims COVID claims mentions in the paper can be found at [covid/claims.txt](covid/claims.txt).

## Citation

```bibtex
@inproceedings{Wadden2020FactOF,
  title={Fact or Fiction: Verifying Scientific Claims},
  author={David Wadden and Shanchuan Lin and Kyle Lo and Lucy Lu Wang and Madeleine van Zuylen and Arman Cohan and Hannaneh Hajishirzi},
  booktitle={EMNLP},
  year={2020},
}
```

## Contact

Email: `davidw@allenai.org`.
