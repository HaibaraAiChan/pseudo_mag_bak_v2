import argparse
import time
import numpy as np
import random
import torch
# from torch._C import T
import torch.nn as nn
import dgl
import dgl.function as fn
import torch.nn.functional as F
from load_graph import load_mag
from graphsage_model import SAGE
from statistics import mean
from load_graph import load_reddit, inductive_split, load_ogb, load_cora, load_karate, prepare_data, 



def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.gpu >= 0:
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


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


def prepare_data(device, args):
    """
    Load dataset and compute neighbor-averaged node features used by SIGN model
    """
    data = load_mag(args.dataset, device)
    g, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data
    in_feats_size = g.ndata['feat'].shape[1]
    feats = neighbor_average_features(g, args)
    labels = labels.to(device)
    # move to device
    train_nid = train_nid.to(device)
    val_nid = val_nid.to(device)
    test_nid = test_nid.to(device)

    print('in_feats')
    print(in_feats_size)
    print('feats')
    print(len(feats))
    print(feats[0].size())
    sub_graph = dgl.node_subgraph(g,g.ndata["target_mask"])




    return sub_graph, feats, labels, in_feats_size, n_classes, \
        train_nid, val_nid, test_nid, evaluator


def load_blocks_subtensor(g, feats, labels, blocks, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	
	batch_inputs = feats[0][blocks[0].srcdata[dgl.NID].tolist()].to(device)
	# print('blocks[-1].dstdata')
	# print(blocks[-1].dstdata['_ID'])
	# print('---------------------------------------------------------------------------------------------------')
	batch_labels = labels[blocks[-1].dstdata['_ID']].to(device)
	# print('\nblocks[-1].dstdata')
	# print(len(blocks[-1].dstdata[dgl.NID].tolist()))
	# print('\nblocks[-1].srcdata dgl.NID list')
	# print(len(blocks[-1].srcdata[dgl.NID].tolist()))
	# print(blocks[-1].srcdata[dgl.NID])
	# print('batch input')
	# print(len(batch_inputs.tolist()))
   
	# print('batch_inputs device')
	# print(batch_inputs.device)
	return batch_inputs, batch_labels, len(blocks[-1].srcdata[dgl.NID].tolist())


def compute_acc(pred, labels):
	"""
	Compute the accuracy of prediction given the labels.
	"""
	labels = labels.long()
	return (torch.argmax(pred, dim=1)==labels).float().sum() / len(pred)


def train(model, train_g, feats, train_labels, loss_fcn, optimizer, train_loader):
    avg_step_data_trans_time_list = []
    avg_step_GPU_train_time_list = []
    tttt = time.time()
    total_train_time=[]

    
    model.train()
    device = train_labels.device

    step_time_list = []
    step_data_trans_time_list = []
    step_GPU_train_time_list = []

    total_nodes=[]
    total_edges=[]

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    t____ = time.time()

    for step, (src, output, blocks )in enumerate (train_loader):
        time_start = time.time()
       
        torch.cuda.synchronize()
        start.record()
        #---------------------------------------------------------------------------------------------$$$$$$$$
        batch_inputs, batch_labels, num_src_nodes = load_blocks_subtensor(train_g, feats, train_labels, blocks, device)
        blocks = [block.int().to(device) for block in blocks]
        #---------------------------------------------------------------------------------------------#######
        end.record()
        torch.cuda.synchronize()  # wait for move to complete
        step_data_trans_time_list.append(start.elapsed_time(end))

        total_nodes.append(num_src_nodes)

        #----------------------------------------------------------
        start1 = torch.cuda.Event(enable_timing=True)
        end1 = torch.cuda.Event(enable_timing=True)
        start1.record()
        #---------------------------------------------------------------------------------------------$$$$$$$
        batch_pred = model(blocks, batch_inputs)
        pseudo_mini_loss = loss_fcn(batch_pred, batch_labels)
        # loss = loss_fcn(model(batch_feats), labels[batch])
        optimizer.zero_grad()
        pseudo_mini_loss.backward()
        optimizer.step()
        #---------------------------------------------------------------------------------------------######
        end1.record()
        torch.cuda.synchronize()  # wait for all training steps to complete
        step_GPU_train_time_list.append(start1.elapsed_time(end1))


        time_4_current_batch= time.time()-time_start
        total_train_time.append(time_4_current_batch)

    totoal_time = time.time()-t____   
    print("train nodes number of this epoch ", sum(total_nodes))
        # acc = compute_acc(batch_pred, batch_labels)

        # gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1024 / 1024 /1024 if torch.cuda.is_available() else 0
        # print(
		# 			'Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.4f} GB'.format(
		# 				0, 0, pseudo_mini_loss.item(), acc.item(), np.mean(0), gpu_mem_alloc))
    print('------------------------------------------------------------------time -------------------------------train ', time.time()-tttt)
    # indent = 0 # skip the first 2 epochs, initial epoch time is not stable.
    # print('time for training batches')
    # print('num of batches ',len(total_train_time))
    # print('avg batch training time ',mean(total_train_time[1:]))
    
    
    # print('avg batch training CPU time ',mean(total_train_time[1:]))
    # avg_iteration_time = sum(step_data_trans_time_list[indent:]) / len(step_data_trans_time_list[indent:])
    # print('\t\tavg iteration(step) data from cpu to GPU time:%.8f sec' % ((avg_iteration_time)/1000))
    # avg_iteration_gpu_time = sum(step_GPU_train_time_list[indent:]) / len(step_GPU_train_time_list[indent:])
    # print('\t\tavg iteration GPU training time:%.8f sec' % ((avg_iteration_gpu_time)/1000))
    # print('total time of current train function CPU',sum(total_train_time))
    

    # return totoal_time, sum(total_train_time), sum(step_data_trans_time_list), sum(step_GPU_train_time_list) 
    return totoal_time, sum(total_train_time), sum(step_data_trans_time_list), sum(step_GPU_train_time_list) , sum(total_nodes)




			
def evaluate(model, g, nfeat, labels, val_nid, device):
	"""
	Evaluate the model on the validation set specified by ``val_nid``.
	g : The entire graph.
	inputs : The features of all the nodes.
	labels : The labels of all the nodes.
	val_nid : the node Ids for validation.
	device : The GPU device to evaluate on.
	"""
	model.eval()
	with torch.no_grad():
		pred = model.inference(g, nfeat, device, args)
	model.train()
	return compute_acc(pred[val_nid], labels[val_nid].to(pred.device))

# def test(model, feats, labels, test_loader, evaluator,
#          train_nid, val_nid, test_nid):
#     model.eval()
#     device = labels.device
#     preds = []
#     for batch in test_loader:
#         batch_feats = [feat[batch].to(device) for feat in feats]
#         preds.append(torch.argmax(model(batch_feats), dim=-1))
#     # Concat mini-batch prediction results along node dimension
#     preds = torch.cat(preds, dim=0)
#     train_res = evaluator(preds[train_nid], labels[train_nid])
#     val_res = evaluator(preds[val_nid], labels[val_nid])
#     test_res = evaluator(preds[test_nid], labels[test_nid])
#     return train_res, val_res, test_res

def pre_process(args, data, device):
    g, feats, labels, in_feats, num_classes, \
        train_nid, val_nid, test_nid, evaluator = data
    # train_nid = torch.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
	# val_nid = torch.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
	# test_nid = torch.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]
    # dataloader_device = torch.device('cpu')

    # train_loader = torch.utils.data.DataLoader(
    #     train_nid, batch_size=args.batch_size, shuffle=True, drop_last=False)
    train_g = val_g = test_g = g
    train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('feat')
    print('train_nfeat')
    print(train_nfeat)
    print(train_nfeat.size())
    print('labels')
    print(labels)
    print(len(labels))
    print("g.ndata")
    # print(g.ndata)
    print(g.ndata['_ID'])
    # print(len(g.ndata['_ID']))
    # print(g.ndata['_TYPE'])
    # print(len(g.ndata['_TYPE']))
    
    # print(len(g.ndata['target_mask']))
    
    # train_labels = val_labels = test_labels = g.ndata.pop('label')
    train_labels = val_labels = test_labels = labels

    print(g.ndata)
    if not args.data_cpu:
        train_nfeat = train_nfeat.to(device)
        train_labels = train_labels.to(device)
	# get_memory("-----------------------------------------after label***************************")
	
	# Create csr/coo/csc formats before launching training processes with multi-gpu.
	# This avoids creating certain formats in each sub-process, which saves momory and CPU.
    train_g.create_formats_()
    val_g.create_formats_()
    test_g.create_formats_()

    tmp = (train_g.in_degrees()==0) & (train_g.out_degrees()==0)
    isolated_nodes = torch.squeeze(torch.nonzero(tmp, as_tuple=False))
    train_g.remove_nodes(isolated_nodes)

    return train_g, train_nid,in_feats,num_classes, feats, train_labels, test_g, test_nid, test_labels



def run(args, data, device):

    train_g, train_nid,in_feats,num_classes, feats, train_labels, test_g, test_nid, test_labels= data
   
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])

    # full_batch_size = len(train_nid)
    full_batch_train_dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        batch_size=args.batch_size,

        # batch_size=full_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # test_loader = torch.utils.data.DataLoader(
    #     torch.arange(test_labels.shape[0]), batch_size=args.eval_batch_size,
    #     shuffle=False, drop_last=False)

    # Initialize model and optimizer for each run
    # num_hops = args.R + 1
    model = SAGE(in_feats, args.num_hidden, num_classes, args.num_layers, F.relu, args.dropout, args.aggre)
    # model = SAGE(in_size, args.num_hidden, num_classes, num_hops, args.ff_layer, args.dropout, args.input_dropout)
    model = model.to(device)
    # print("# Params:", get_n_params(model))

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
    #                                 weight_decay=args.weight_decay)

    # Start training
    best_epoch = 0
    best_val = 0
    best_test = 0
    time_list=[]
    avg_step_data_trans_time_list = []
    avg_step_GPU_train_time_list = []
    avg_step_time_list = []

    nodes_epoch=[]

    ttt = []
    for epoch in range(1, args.num_epochs + 1):
        print('-'*80+'start epoch ', epoch)
        start = time.time()
        # t___, t0, t1, t2= train(model, train_g, feats, train_labels, loss_fcn, optimizer, full_batch_train_dataloader)
        
        t___, t0, t1, t2, nodes_num= train(model, train_g, feats, train_labels, loss_fcn, optimizer, full_batch_train_dataloader)
        time_list.append(time.time()-start)
        nodes_epoch.append(nodes_num)
        ttt.append(t___)
        avg_step_time_list.append(t0)
        avg_step_data_trans_time_list.append(t1)
        avg_step_GPU_train_time_list.append(t2)
    out_indent=2
    total_avg_iteration_time = sum(avg_step_data_trans_time_list[out_indent:]) / len(avg_step_data_trans_time_list[out_indent:])
    print('\ttotal avg iteration(step) data from cpu to GPU time:%.8f s' % (total_avg_iteration_time/1000))
    # print(len(avg_step_data_trans_time_list))

    total_avg_iteration_gpu_time = sum(avg_step_GPU_train_time_list[out_indent:]) / len(avg_step_GPU_train_time_list[out_indent:])
    print('\ttotal avg iteration GPU training time:%.8f s' % (total_avg_iteration_gpu_time/1000))
    # avg_step_time = sum(step_time_list[indent:]) / len(step_time_list[indent:])
    total_avg_step_time = sum(avg_step_time_list[out_indent:]) / len(avg_step_time_list[out_indent:])
    print('\ttotal avg iteration (step) total CPU time:%.8f s' % (total_avg_step_time ))

    total_avg_epoch_time = sum(ttt[out_indent:]) / len(ttt[out_indent:])
    print('\ttotal avg epoch total CPU time including train loader :%.8f s' % (total_avg_epoch_time ))


    avg_epoch_nodes = sum(nodes_epoch) / len(nodes_epoch)
    print('\tavg epoch nodes input :%.8f s' % (avg_epoch_nodes ))

        
    print('-'*80+'end training')
    print('time for training epoches')
    # print('time for 6 epoch',time_list)
    print('num of epoches ',len(time_list))
    print('avg epoches training time ',mean(time_list[1:]))

    # print('time for 6 epoch',sum(time_list[1:]))



    return best_test


def main(args):
    device = "cpu"
	if args.dataset=='karate':
		g, n_classes = load_karate()
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='cora':
		g, n_classes = load_cora()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='reddit':
		g, n_classes = load_reddit()
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
		print('#nodes:', g.number_of_nodes())
		print('#edges:', g.number_of_edges())
		print('#classes:', n_classes)
	elif args.dataset=='ogbn-products':
		g, n_classes = load_ogb(args.dataset)
		# print('#nodes:', g.number_of_nodes())
		# print('#edges:', g.number_of_edges())
		# print('#classes:', n_classes)
		device = "cuda:0"
		data=prepare_data(g, n_classes, args, device)
	elif args.dataset=='ogbn-mag':
		device = "cpu"
        data = prepare_data(device, args)
        device = "cuda:0"
        data = pre_process(args, data, device)
		
	else:
		raise Exception('unknown dataset')
	
	best_test = run(args, device, data)


    if args.gpu < 0:
        device = "cpu"
    else:
        device = "cuda:{}".format(args.gpu)

    with torch.no_grad():
        device = "cpu"
        data = prepare_data(device, args)
        device = "cuda:0"
        data = pre_process(args, data, device)
    val_accs = []
    test_accs = []

    best_test = run(args, data, device)



    # for i in range(args.num_epochs):
    #     print(f"Run {i} start training")
    #     best_test = run(args, data, device)
    #     print('test_acc')
    #     print(best_test)
        # best_val, best_test = run(args, data, device)
        # val_accs.append(best_val)
        # test_accs.append(best_test)

    # print(f"Average val accuracy: {np.mean(val_accs):.4f}, "
    #       f"std: {np.std(val_accs):.4f}")
    # print(f"Average test accuracy: {np.mean(test_accs):.4f}, "
    #       f"std: {np.std(test_accs):.4f}")


if __name__ == "__main__":
    tt = time.time()
    print("main start at this time " + str(tt))
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
		help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--seed', type=int, default=1236)

    argparser.add_argument('--dataset', type=str, default='ogbn-mag')
    # argparser.add_argument('--dataset', type=str, default='ogbn-products')
    argparser.add_argument('--aggre', type=str, default='lstm')
    # argparser.add_argument('--dataset', type=str, default='cora')
    # argparser.add_argument('--dataset', type=str, default='karate')
    # argparser.add_argument('--dataset', type=str, default='reddit')
    # argparser.add_argument('--aggre', type=str, default='mean')
    argparser.add_argument('--selection-method', type=str, default='range')
    argparser.add_argument('--num-epochs', type=int, default=3)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=1)
    # argparser.add_argument('--fan-out', type=str, default='20')
    argparser.add_argument('--fan-out', type=str, default='10')

    # argparser.add_argument('--batch-size', type=int, default=157393)
    argparser.add_argument('--batch-size', type=int, default=78697)
    # argparser.add_argument('--batch-size', type=int, default=39349)
    # argparser.add_argument('--batch-size', type=int, default=19675)
    # argparser.add_argument('--batch-size', type=int, default=9838)
    # argparser.add_argument('--batch-size', type=int, default=4919)



    # argparser.add_argument('--batch-size', type=int, default=196571)
    # argparser.add_argument('--batch-size', type=int, default=98308)
    # argparser.add_argument('--batch-size', type=int, default=49154)
    # argparser.add_argument('--batch-size', type=int, default=24577)
    # argparser.add_argument('--batch-size', type=int, default=12289)
    # argparser.add_argument('--batch-size', type=int, default=6145)
    # argparser.add_argument('--batch-size', type=int, default=3000)
    # argparser.add_argument('--batch-size', type=int, default=500)
    # argparser.add_argument('--batch-size', type=int, default=8)
    argparser.add_argument("--eval-batch-size", type=int, default=100000,
                        help="evaluation batch size")
    argparser.add_argument("--R", type=int, default=5,
                        help="number of hops")


    argparser.add_argument('--log-every', type=int, default=5)
    argparser.add_argument('--eval-every', type=int, default=5)

    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument("--weight-decay", type=float, default=0)
    argparser.add_argument('--num-workers', type=int, default=4,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
        help="Inductive learning setting")
    argparser.add_argument('--data-cpu', action='store_true',
        help="By default the script puts all node features and labels "
                "on GPU when using it to save time for data copy. This may "
                "be undesired if they cannot fit in GPU memory at once. "
                "This flag disables that.")
    args = argparser.parse_args()
    set_seed(args)

    print(args)
    main(args)
