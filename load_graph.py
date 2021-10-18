# import argparse
# import time
# import numpy as np
import torch
# import torch.nn as nn
# import dgl
# import dgl.function as fn
# from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

# import torch.nn.functional as F
# import gc

import dgl
import torch as th
import dgl.function as fn
from cpu_mem_usage import get_memory
import time
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

def get_ogb_evaluator(dataset):
    """
    Get evaluator from Open Graph Benchmark based on dataset
    """
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
        "y_true": labels.view(-1, 1),
        "y_pred": preds.view(-1, 1),
    })["acc"]


def convert_mag_to_homograph(g):
    """
    Featurize node types that don't have input features (i.e. author,
    institution, field_of_study) by averaging their neighbor features.
    Then convert the graph to a undirected homogeneous graph.
    """
    src_writes, dst_writes = g.all_edges(etype="writes")
    src_topic, dst_topic = g.all_edges(etype="has_topic")
    src_aff, dst_aff = g.all_edges(etype="affiliated_with")
    new_g = dgl.heterograph({
        ("paper", "written", "author"): (dst_writes, src_writes),
        ("paper", "has_topic", "field"): (src_topic, dst_topic),
        ("author", "aff", "inst"): (src_aff, dst_aff)
    })
    # new_g = new_g.to(device)
    new_g.nodes["paper"].data["feat"] = g.nodes["paper"].data["feat"]

    new_g["written"].update_all(fn.copy_u("feat", "m"), fn.mean("m", "feat"))
    new_g["has_topic"].update_all(fn.copy_u("feat", "m"), fn.mean("m", "feat"))
    new_g["aff"].update_all(fn.copy_u("feat", "m"), fn.mean("m", "feat"))

    g.nodes["author"].data["feat"] = new_g.nodes["author"].data["feat"]
    g.nodes["institution"].data["feat"] = new_g.nodes["inst"].data["feat"]
    g.nodes["field_of_study"].data["feat"] = new_g.nodes["field"].data["feat"]


    # Convert to homogeneous graph
    # Get DGL type id for paper type
    target_type_id = g.get_ntype_id("paper")
    print('target_type_id ',target_type_id)
    g = dgl.to_homogeneous(g, ndata=["feat"])
    g = dgl.add_reverse_edges(g, copy_ndata=True)
    # Mask for paper nodes
    g.ndata["target_mask"] = g.ndata[dgl.NTYPE] == target_type_id
    output, counts = th.unique_consecutive(g.ndata[dgl.NTYPE], return_counts=True)
    print('counts',counts)
    
    return g

def neighbor_average_features(g, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats")
    g.ndata["feat_0"] = g.ndata["feat"]
    for hop in range(1, args.R + 1):
        g.update_all(fn.copy_u(f"feat_{hop-1}", "msg"),
                     fn.mean("msg", f"feat_{hop}"))
    res = []
    for hop in range(args.R + 1):
        res.append(g.ndata.pop(f"feat_{hop}"))

    if args.dataset == "ogbn-mag":
        # For MAG dataset, only return features for target node types (i.e.
        # paper nodes)
        target_mask = g.ndata["target_mask"]
        target_ids = g.ndata[dgl.NID][target_mask]
        num_target = target_mask.sum().item()
        new_res = []
        for x in res:
            feat = torch.zeros((num_target,) + x.shape[1:],
                               dtype=x.dtype, device=x.device)
            feat[target_ids] = x[target_mask]
            new_res.append(feat)
        res = new_res
    return res

    
def prepare_data(g, n_classes, args, device):
    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
        train_nfeat = train_g.ndata.pop('features')
        val_nfeat = val_g.ndata.pop('features')
        test_nfeat = test_g.ndata.pop('features')
        train_labels = train_g.ndata.pop('labels')
        val_labels = val_g.ndata.pop('labels')
        test_labels = test_g.ndata.pop('labels')
    else:
        train_g = val_g = test_g = g
        train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('feat')
        train_labels = val_labels = test_labels = g.ndata.pop('label')

    # get_memory("-----------------------------------------after inductive else***************************")
    # t4 = ttt(t3, "after inductive else")
    # print('args.data_cpu')
    # print(args.data_cpu)

    if not args.data_cpu:
        train_nfeat = train_nfeat.to(device)
        train_labels = train_labels.to(device)
    # get_memory("-----------------------------------------after label***************************")
    # t5 = ttt(t4, "after label")
    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves momory and CPU.
    train_g.create_formats_()
    # get_memory("-----------------------------------------after  train_g.create_formats_()***************************")
    val_g.create_formats_()
    # get_memory("-----------------------------------------after  train_g.create_formats_()***************************")
    test_g.create_formats_()
    # get_memory("-----------------------------------------before pack data***************************")
    # t6 = ttt(t5, "after train_g.create_formats_()")
    # see_memory_usage("-----------------------------------------after model to gpu------------------------")
    tmp = (train_g.in_degrees()==0) & (train_g.out_degrees()==0)
    isolated_nodes = torch.squeeze(torch.nonzero(tmp, as_tuple=False))
    train_g.remove_nodes(isolated_nodes)
    # Pack data
    data = n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
        val_nfeat, val_labels, test_nfeat, test_labels
    return data

def prepare_data_mag(device, args):
    """
    Load dataset and compute neighbor-averaged node features used by SIGN model
    """
    data = load_mag(args.dataset, device)
    total_g, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data

    in_feats = total_g.ndata['feat'].shape[1]
    feats = neighbor_average_features(total_g, args)
    labels = labels.to(device)
    # move to device
    train_nid = train_nid.to(device)
    val_nid = val_nid.to(device)
    test_nid = test_nid.to(device)

    sub_graph = dgl.node_subgraph(total_g,total_g.ndata["target_mask"])
    print(type(sub_graph))
    print('g.ndata[feat]')
    print(type(total_g.ndata['feat']))
    print(total_g.ndata['feat'])
    print('total graph length of feats')
    print(len(feats))

    sub_graph.ndata['features']= feats[0] # we only keep the 1-hop neighbor mean feature value
    # print('sub graph len(feats[0])')
    # print(len(feats[0]))
    # print(sub_graph.ndata['features'])
    # print('len(subgraph.ndata[nid])')
    # print(len(sub_graph.ndata['_ID']))
    # print(sub_graph.ndata)
	
    print('*'*90)
    

    train_mask = torch.zeros((sub_graph.number_of_nodes(),), dtype=torch.bool)
    train_mask[train_nid] = True
    val_mask = torch.zeros((sub_graph.number_of_nodes(),), dtype=torch.bool)
    val_mask[val_nid] = True
    test_mask = torch.zeros((sub_graph.number_of_nodes(),), dtype=torch.bool)
    test_mask[test_nid] = True
    sub_graph.ndata['train_mask'] = train_mask
    sub_graph.ndata['val_mask'] = val_mask
    sub_graph.ndata['test_mask'] = test_mask
    print('sub_graph.ndata["label"]')
    print(labels)
    print(len(labels))
    sub_graph.ndata['labels'] = sub_graph.ndata['label']=labels #---------------------------------------------------------

    tmp = (sub_graph.in_degrees()==0) & (sub_graph.out_degrees()==0)
    isolated_nodes = torch.squeeze(torch.nonzero(tmp, as_tuple=False))
    sub_graph.remove_nodes(isolated_nodes)

    train_g = val_g = test_g = sub_graph
    train_labels = val_labels = test_labels = labels
    train_nfeat = val_nfeat = test_nfeat = sub_graph.ndata['feat']
    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()
    # print('------------------------------------------------------------------------train_g')
    # print(train_g)
    # print(train_g.ndata['feat'])
    # print(len(train_g.ndata['feat']))
    # print(train_g.ndata['features'])
    # print(len(train_g.ndata['features']))

    

    # return train_g, labels, in_feats, n_classes, train_nid, test_nid
	# return sub_graph, labels, in_feats, n_classes, train_nid, val_nid, test_nid
    return n_classes, train_g, val_g, test_g, train_nfeat, train_labels, val_nfeat, val_labels, test_nfeat, test_labels

def load_mag(name,device):

    tic_step = time.time()
    get_memory("-" * 40 + "---------------------from ogb.nodeproppred import DglNodePropPredDataset***************************")
    print('load', name)
    dataset = DglNodePropPredDataset(name=name)
    t1 = ttt(tic_step, "-"*40+"---------------------data = DglNodePropPredDataset(name=name)***************************\n")
    # get_memory("-"*40+"---------------------data = DglNodePropPredDataset(name=name)***************************")
    print('finish loading', name)
   
    g, labels = dataset[0]
    total_g = convert_mag_to_homograph(g)
   
    paper_labels = labels['paper'].squeeze()
    print('len(paper label)')
    print(len(paper_labels))
    split_idx = dataset.get_idx_split()
    print(split_idx)
    train_nid = split_idx["train"]['paper']
    val_nid = split_idx["valid"]['paper']
    test_nid = split_idx["test"]['paper']
   

    num_classes = dataset.num_classes
    evaluator = get_ogb_evaluator(name)
    # print(g)
    # print(len(g.ndata["_ID"]))
    # print(len(g.ndata["_TYPE"]))
    # print(len(g.ndata["target_mask"]))
    
    print(f"# total Nodes: {total_g.number_of_nodes()}\n"
          f"# total Edges: {total_g.number_of_edges()}\n"
          f"# paper graph Labels: {len(paper_labels)}\n"
          f"# paper graph Train: {len(train_nid)}\n"
          f"# paper graph Val: {len(val_nid)}\n"
          f"# paper graph Test: {len(test_nid)}\n"
          f"# paper graph Classes: {num_classes}")

    return total_g, paper_labels, num_classes, train_nid, val_nid, test_nid, evaluator
    # return g, num_classes





def ttt(tic, str1):
    toc = time.time()
    print(str1 + '\n step Time(s): {:.4f}'.format(toc - tic))
    return toc


def load_ogbn_dataset(name,  args):
    """
    Load dataset and move graph and features
    """
    '''if name not in ["ogbn-products", "ogbn-arxiv","ogbn-mag"]:
        raise RuntimeError("Dataset {} is not supported".format(name))'''
    if name not in ["ogbn-products", "ogbn-mag","ogbn-papers100M"]:
        raise RuntimeError("Dataset {} is not supported".format(name))
    dataset = DglNodePropPredDataset(name=name, root = args.root)
    splitted_idx = dataset.get_idx_split()
    print(name)

    if name=="ogbn-papers100M":
        train_nid = splitted_idx["train"]
        val_nid = splitted_idx["valid"]
        test_nid = splitted_idx["test"]
        g, labels = dataset[0]        
        n_classes = dataset.num_classes        
        labels = labels.squeeze()
        evaluator = get_ogb_evaluator(name)        
        print(f"# Nodes: {g.number_of_nodes()}\n"
            f"# Edges: {g.number_of_edges()}\n"
            f"# Train: {len(train_nid)}\n"
            f"# Val: {len(val_nid)}\n"
            f"# Test: {len(test_nid)}\n"
            f"# Classes: {n_classes}\n")

        return g, labels, n_classes, train_nid, val_nid, test_nid, evaluator




def load_karate():
    from dgl.data import KarateClubDataset

    # load reddit data
    data = KarateClubDataset()
    g = data[0]
    print('karate data')
    # print(data[0].ndata)
    # print(data[0].edata)
    ndata=[]
    for nid in range(34):
        ndata.append((th.ones(4)*nid).tolist())
    ddd = {'feat': th.tensor(ndata)}
    g.ndata['features'] = ddd['feat']
    g.ndata['feat'] = ddd['feat']
    # print(data[0].ndata)
    g.ndata['labels'] = g.ndata['label']

    train = [True]*24 + [False]*10
    val = [False] * 24 + [True] * 5 + [False] * 5
    test = [False] * 24 + [False] * 5 + [True] * 5
    g.ndata['train_mask'] = th.tensor(train)
    g.ndata['val_mask'] = th.tensor(val)
    g.ndata['test_mask'] = th.tensor(test)

    return g, data.num_classes


def load_cora():
    from dgl.data import CoraGraphDataset

    # load reddit data
    data = CoraGraphDataset()
    g = data[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    return g, data.num_classes


def load_reddit():
    from dgl.data import RedditDataset

    # load reddit data
    data = RedditDataset(self_loop=True)
    g = data[0]
    g = dgl.remove_self_loop(g)
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    return g, data.num_classes

def load_ogb(name):

    tic_step = time.time()
    get_memory("-" * 40 + "---------------------from ogb.nodeproppred import DglNodePropPredDataset***************************")
    print('load', name)
    data = DglNodePropPredDataset(name=name)
    t1 = ttt(tic_step, "-"*40+"---------------------data = DglNodePropPredDataset(name=name)***************************")
    # get_memory("-"*40+"---------------------data = DglNodePropPredDataset(name=name)***************************")
    print('finish loading', name)
    splitted_idx = data.get_idx_split()
    t2 = ttt(t1,"-" * 40 + "---------------------splitted_idx = data.get_idx_split()***************************")
    # get_memory("-" * 40 + "---------------------splitted_idx = data.get_idx_split()***************************")
    graph, labels = data[0]

    graph = dgl.remove_self_loop(graph) #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    

    # get_memory("-" * 40 + "---------------------graph, labels = data[0]***************************")
    t3 = ttt(t2, "-" * 40 + "---------------------graph, labels = data[0]***************************")
    print(labels)
    print('graph after remove self connected edges')
    print(graph)
    labels = labels[:, 0]
    # get_memory("-" * 40 + "---------------------labels = labels[:, 0]***************************")
    t4 = ttt(t3, "-" * 40 + "---------------------labels = labels[:, 0]***************************")

    graph.ndata['features'] = graph.ndata['feat']
    # get_memory("-" * 40 + "---------------------graph.ndata['features'] = graph.ndata['feat']***************************")
    t5 = ttt(t4, "-" * 40 + "---------------------graph.ndata['features'] = graph.ndata['feat']***************************")
    graph.ndata['labels'] = labels
    graph.ndata['label'] = labels
    t6 = ttt(t5, "-" * 40 + "---------graph.ndata['labels'] = labels******************")
    in_feats = graph.ndata['features'].shape[1]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    t7 = ttt(t6, "-" * 40 + "---------train_nid, val_nid, test_nid = splitted_idx******************")
    # get_memory(
	    # "-" * 40 + "---------------------train_nid, val_nid, test_nid = splitted_idx***************************")
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask
    t8 = ttt(t7, "-" * 40 + "---------end of load ogb******************")
    # get_memory(
	    # "-" * 40 + "---------------------end of load ogb***************************")

    print('finish constructing', name)
    print('load ogb-products time total: '+ str(time.time()-tic_step))
    return graph, num_labels

def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['train_mask'] | g.ndata['val_mask'])
    test_g = g
    return train_g, val_g, test_g
