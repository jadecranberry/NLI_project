# NLI_project
This repository contains code for the Postdoctoral Fellowship project at Factually Health Inc.

## Table of contents
- [Introduction](#Introduction)
- [Dependencies](#dependencies)
- [Run models for paper metrics](#run-models-for-paper-metrics)
- [Dataset](#dataset)
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



## Dependencies

We recommend you create an anaconda environment:
```bash
conda create --name scifact python=3.7 conda-build
```
Then, from the `scifact` project root, run
```
conda develop .
```
which will add the scifact code to your `PYTHONPATH`.

Then, install Python requirements:
```
pip install -r requirements.txt
```

## Run models for paper metrics

We provide scripts let you easily run our models and re-create the metrics published in paper. The script will automatically download the dataset and pre-trained models. You should be able to reproduce our dev set results from the paper by following these instructions (we are not releasing test set labels at this point). Please post an issue if you're unable to do this.

To recreate Table 3 rationale selection metrics:
```bash
./script/rationale-selection.sh [bert-variant] [training-dataset] [dataset]
```
To recreate Table 3 label prediction metrics:
```bash
./script/label-prediction.sh [bert-variant] [training-dataset] [dataset]
```
- `[bert-variant]` options: `roberta_large`, `roberta_base`, `scibert`, `biomed_roberta_base`
- `[training-dataset]` options: `scifact`, `scifact_only_claim`, `scifact_only_rationale`, `fever_scifact`, `fever`, `snopes`
- `[dataset]` options: `dev`, `test`

To recreate Table 4:
```bash
./script/pipeline.sh [retrieval] [model] [dataset]
```
- `[retrieval]` options: `oracle`, `open`
- `[model]` options: `oracle-rationale`, `zero-shot`, `verisci`
- `[dataset]` options: `dev`, `test`


## Dataset

Download with script: The data will be downloaded and stored in the `data` directory.
```bash
./script/download-data.sh
```
Or, [click here](https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz) to download the tarball.

The claims are split into `claims_train.jsonl`, `claims_dev.jsonl`, and `claims_test.jsonl`, one claim per line. The claim and dev sets contain labels, while the test set is unlabeled. For test set evaluation, submit to the [leaderboard](https://leaderboard.allenai.org/scifact)! The corpus of evidence documents is `corpus.jsonl`, one evidence document per line.

Due to the relatively small size of the dataset, we also provide a 5-fold cross-validation split that may be useful for model development. After unzipping the tarball, the data will organized like this:

```
data
| corpus.jsonl
| claims_train.jsonl
| claims_dev.jsonl
| claims_test.jsonl
| cross_validation
  | fold_1
    | claims_train_1.jsonl
    | claims_dev_1.jsonl
  ...
  | fold_5
    | claims_train_5.jsonl
    | claims_dev_5.jsonl
```

See [data.md](doc/data.md) for descriptions of the schemas for each file type.


### Claim generation data

We also make available the collection of claims together with the documents and citation contexts they are based on. We hope that these data will facilitate the training of "claim generation" models that can summarize a citation context into atomic claims. Click [here](https://scifact.s3-us-west-2.amazonaws.com/release/latest/claims_with_citances.jsonl) to download the file, or enter

```bash
wget https://scifact.s3-us-west-2.amazonaws.com/release/latest/claims_with_citances.jsonl -P data
```

For more information on the data, see [claims-with-citances.md](doc/claims-with-citances.md)


## Download pre-trained models

All "BERT-to-BERT"-style models as described in the paper are stored in a public AWS S3 bucket. You can download the models models using the script:
```bash
./script/download-model.sh [model-component] [bert-variant] [training-dataset]
```
- `[model-component]` options: `rationale`, `label`
- `[bert-variant]` options: `roberta_large`, `roberta_base`, `scibert`, `biomed_roberta_base`
- `[training-dataset]` options: `scifact`, `scifact_only_claim`, `scifact_only_rationale`, `fever_scifact`, `fever`, `snopes`

The script checks to make sure the downloaded model doesn't already exist before starting new downloads.

The best-performing pipeline reported in [paper](https://arxiv.org/abs/2004.14974) uses:
- `rationale`: `roberta_large` + `scifact`
- `label`: `roberta_large` + `fever_scifact`

For `fever` and `fever_scifact`, there are models available for all 4 BERT variants. For `snopes`, only `roberta_large` is available for download (but you can train your own model).

After downloading the pretrained-model, you can follow instruction [model.md](doc/model.md) to run individual model components.

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
