import argparse
from subprocess import call
import os
import torch

from verisci.covid import AbstractRetriever, EmbeddingGenerator


def get_args():
    parser = argparse.ArgumentParser(
        description="Verify a claim against the CORD-19 corpus.")
    parser.add_argument("--n_documents", type=int, default=20,
                        help="The number of documents to retrieve from Covidex.")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose model output.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use. Defaults to `gpu` if one is available, else `cpu`.")
    parser.add_argument("--rationale_model", type=str, default="rationale_roberta_large_scifact",
                        help="name of the directory under './model' containing the model to be loaded. If the directory does not exist, the script will attempt to find a mtach on the HuggingFace catalog and download it.")
    parser.add_argument("--tf", type=bool, default=False,
                        help="True if this is a tensorflow model")
    return parser.parse_args()


def inference(args):
    # Initialize pipeline components
    if args.verbose:
        print("Initializing model.")
    rationale_selection_model = args.rationale_model
    if os.path.isdir('model/' + args.rationale_model):
        rationale_selection_model = 'model/' + rationale_selection_model
    

    # Get device.
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    abstract_retriever = AbstractRetriever()
    embedding_generator = EmbeddingGenerator(rationale_selection_model,
                                           device, tf=args.tf)

    # Run model.
    if args.verbose:
        print("Retrieving abstracts.")
    results = abstract_retriever("", k=args.n_documents)

    if args.verbose:
        print("Generating Embeddings")
    results = embedding_generator(results)


def main():
    args = get_args()
    results = inference(args)

    print("Done!")

if __name__ == "__main__":
    main()
