import os
import pandas as pd
import re
def merge_datasets(generation_dataset:str, solutions_dataset:str, output:str=None, threads:int=None):
    """
    Merge the datasets in the graphs_path directory into a single dataset and save it to the output_path directory.
    """
    generation_dataset = pd.read_csv(generation_dataset)
    solutions_dataset = pd.read_json(solutions_dataset)
    solutions_dataset["filename"] = solutions_dataset["graph"].apply(lambda x:x.rstrip(".weighted_edgelist.gz"))
    # Merge the datasets
    dataset = pd.merge(generation_dataset, solutions_dataset, on="filename")
    # Save the dataset
    if output is not None:
        dataset.to_csv(output)
    return dataset

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--generation-dataset",required=True,type=str,default="dataset",help="The dataset containing the generation results")
    parser.add_argument("--solutions-dataset",required=True,type=str,default="dataset",help="The dataset containing the solutions")
    parser.add_argument("--output",type=str,default="dataset",help="The output dataset")
    args=parser.parse_args()
    merge_datasets(args.generation_dataset, args.solutions_dataset, args.output)