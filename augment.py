import numpy as np
import pandas as pd
import os
import mpg.mpgio as mpgio
import networkx as nx
import itertools

def preprocess_dataset(graphs_path:str,dataset_path:str,augmentation="all",output:str=None):
    graphs=os.listdir(graphs_path)
    dataset=pd.read_csv(dataset_path)
    graph_files=set(dataset["graph"].unique())
    dataset.set_index("graph",inplace=True)
    dataset["winners_min"]=dataset["winners_min"].apply(lambda x:np.array(x[1:-1].split(",")).astype(int))
    dataset["winners_max"]=dataset["winners_max"].apply(lambda x:np.array(x[1:-1].split(",")).astype(int))
    final_dataset=[]
    for graph_file in os.listdir(graphs_path):
        if graph_file not in graph_files:
            continue
        graph=mpgio.read_mpg(os.path.join(graphs_path,graph_file))
        row=dataset.loc[graph_file]
        adjacency_data={}
        adjmatrix=nx.adjacency_matrix(graph,weight=None)
        weightsmatrix=nx.adjacency_matrix(graph,weight="weight")
        for i,j in itertools.product(*map(range,adjmatrix.shape)):
            adjacency_data[f"M{(i,j)}"]=adjmatrix[i,j]
            adjacency_data[f"W{(i,j)}"]=weightsmatrix[i,j]
        if augmentation=="all":
            for k,winner in enumerate(row["winners_min"]):
                turn="MIN"
                final_dataset.append(dict(graph=graph, filename=row["filename"],
                                          dataset=row["dataset"],turn=turn,
                                          winner="MAX" if winner==0 else "MIN",
                                          starting_player=turn, starting_vertex=k,
                                          **adjacency_data))
            for k, winner in enumerate(row["winners_max"]):
                turn = "MAX"
                final_dataset.append(dict(graph=graph_file, filename=row["filename"],
                                          dataset=row["dataset"], turn=turn,
                                          winner="MAX" if winner == 0 else "MIN",
                                          starting_player=turn, starting_vertex=k,
                                          **adjacency_data))
    final_dataset= pd.DataFrame(final_dataset)
    if output is not None:
        final_dataset.to_csv(output)
    return final_dataset
if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--graphs-path",required=True,type=str,default="graphs",help="The path to the graphs")
    parser.add_argument("--dataset-path",required=True,type=str,default="dataset",help="The path to the dataset")
    parser.add_argument("--augmentation",type=str,default="all",help="The augmentation to use")
    parser.add_argument("--output",type=str,help="The output dataset")
    args=parser.parse_args()
    preprocess_dataset(args.graphs_path, args.dataset_path, args.augmentation, args.output)

