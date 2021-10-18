import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from block_dataloader import generate_dataloader
from block_dataloader_graph import generate_dataloader
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
import deepspeed
import random
from graphsage_model import SAGE

from load_graph import load_reddit, inductive_split, load_ogb, load_cora, load_karate,  prepare_data, prepare_data_mag
from memory_usage import see_memory_usage
import tracemalloc
from cpu_mem_usage import get_memory
# from utils import draw_graph_global


def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.gpu >= 0:
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True


def ttt(tic, str1):
	toc = time.time()
	print(str1 + ' step Time(s): {:.4f}'.format(toc - tic))
	return toc


def compute_acc(pred, labels):
	"""
	Compute the accuracy of prediction given the labels.
	"""
	labels = labels.long()
	return (torch.argmax(pred, dim=1)==labels).float().sum() / len(pred)

# def intersection_of_2_batches(tensor_a, tensor_b):

# 	indices = torch.zeros_like(tensor_a, dtype = torch.bool)
# 	for elem in tensor_b:
# 	    indices = indices | (tensor_a == elem)  
# 	intersection = tensor_a[indices]  
# 	return intersection

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

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	batch_inputs = nfeat[input_nodes].to(device)
	batch_labels = labels[seeds].to(device)
	print('batch_inputs device')
	print(batch_inputs.device)
	return batch_inputs, batch_labels


def load_blocks_subtensor(g, labels, blocks, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	# print('\nblocks[0].srcdata dgl.NID list')
	# print(blocks[0].srcdata[dgl.NID].tolist())
	batch_inputs = g.ndata['features'][blocks[0].srcdata[dgl.NID].tolist()].to(device)
	batch_labels = blocks[-1].dstdata['labels'].to(device)
	# print('\nblocks[-1].dstdata')
	# print(blocks[-1].dstdata[dgl.NID])
	# print()
	# print('batch_inputs device')
	# print(batch_inputs.device)
	return batch_inputs, batch_labels



#### Entry point
def run(args, device, data):
	# Unpack data
	n_classes, train_g, val_g, test_g, train_nfeat, train_labels, \
	val_nfeat, val_labels, test_nfeat, test_labels = data

	in_feats = train_nfeat.shape[1]
	train_nid = torch.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
	val_nid = torch.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
	test_nid = torch.nonzero(~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]
	dataloader_device = torch.device('cpu')

	sampler = dgl.dataloading.MultiLayerNeighborSampler(
		[int(fanout) for fanout in args.fan_out.split(',')])

	# sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
	full_batch_size = len(train_nid)
	full_batch_dataloader = dgl.dataloading.NodeDataLoader(
		train_g,
		train_nid,
		sampler,

		batch_size=full_batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=args.num_workers)

	model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout, args.aggre)
	model = model.to(device)
	loss_fcn = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=args.lr)

	
	epoch_train_CPU_time_list = []
	# see_memory_usage("-----------------------------------------before for epoch loop ")
	iter_tput = []
	avg_step_data_trans_time_list = []
	avg_step_GPU_train_time_list = []
	avg_step_time_list = []
	block_generate_time_list = []

	full_batch_sub_graph_data_list=[]
	for epoch in range(args.num_epochs):
		print('Epoch ' + str(epoch))
		from dgl.data.utils import load_graphs
		full_batch_subgraph =list(load_graphs('./DATA/'+args.dataset+'_'+str(epoch)+'_subgraph.bin',[0]))
		
		cur_subgraph = full_batch_subgraph[0][0]
		# print(cur_subgraph)
		# print(cur_subgraph.ndata)
		# print(cur_subgraph.edata)
		full_batch_sub_graph_data_list.append(cur_subgraph)


	
	for epoch, full_batch_subgraph in enumerate(full_batch_sub_graph_data_list):
		print('-'*90 + ' Epoch ' + str(epoch))
		
		# Create DataLoader for constructing blocks
		block_dataloader, weights_list, time_collection = generate_dataloader(train_g, full_batch_subgraph,  args)
		print('         full batch subgraph-----------------------------')
		print('full_batch_subgraph.srcdata[''_ID'']')
		print(len(full_batch_subgraph.srcdata['_ID']))
		print('full_batch_subgraph.edata["_ID"]')
		print(len(full_batch_subgraph.edata["_ID"]))
		print(full_batch_subgraph.num_edges())
		full_nodes_num = len(full_batch_subgraph.srcdata['_ID'])
		full_edges_num = len(full_batch_subgraph.edata["_ID"])

		batch_nodes_tensor_list = []
		batch_edges_tensor_list = []
		for step, (input_node, seeds, blocks) in enumerate(block_dataloader):
			# print("\n   ***************************     step   " + str(step) + " mini batch block  **************************")
			print('current block')
			current_block = list(blocks)[0]
			print(current_block)

			num_src = len(list(input_node))
			batch_nodes_tensor_list.append(num_src)
			print(num_src)
			
			edges_num = len(list(current_block.edges())[0])
			batch_edges_tensor_list.append(edges_num)
			print(edges_num)
			# print(current_block.num_nodes())
			# print(current_block.num_edges())
		
		mini_nodes_num = sum(batch_nodes_tensor_list)
		print('pseudo_mini_batch_nodes  sum / full_nodes_num '+ str(mini_nodes_num)+ " / "+str(full_nodes_num) + "  "+ str(mini_nodes_num/full_nodes_num))
		mini_edges_num = sum(batch_edges_tensor_list)
		print('pseudo_mini_batch_edges  sum / full_nodes_num '+ str(mini_edges_num)+ " / "+str(full_edges_num)+ "   "+str(mini_edges_num/full_edges_num))
		
			
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
		data = prepare_data_mag(device, args)
		device = "cuda:0"
		
	else:
		raise Exception('unknown dataset')
	
	best_test = run(args, device, data)


if __name__=='__main__':
	# get_memory("-----------------------------------------main_start***************************")
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
	argparser.add_argument('--num-runs', type=int, default=2)
	argparser.add_argument('--num-epochs', type=int, default=3)
	argparser.add_argument('--num-hidden', type=int, default=16)
	argparser.add_argument('--num-layers', type=int, default=1)
	# argparser.add_argument('--fan-out', type=str, default='20')
	argparser.add_argument('--fan-out', type=str, default='10')


	argparser.add_argument('--batch-size', type=int, default=157393)
	# argparser.add_argument('--batch-size', type=int, default=78697)
	# argparser.add_argument('--batch-size', type=int, default=39349)
	# argparser.add_argument('--batch-size', type=int, default=19675)
	# argparser.add_argument('--batch-size', type=int, default=9838)
	# argparser.add_argument('--batch-size', type=int, default=4919)
	# argparser.add_argument('--batch-size', type=int, default=15341)
	# argparser.add_argument('--batch-size', type=int, default=78697)


	# argparser.add_argument('--batch-size', type=int, default=196571)
	# argparser.add_argument('--batch-size', type=int, default=98308)
	# argparser.add_argument('--batch-size', type=int, default=49154)
	# argparser.add_argument('--batch-size', type=int, default=24577)
	# argparser.add_argument('--batch-size', type=int, default=12289)
	# argparser.add_argument('--batch-size', type=int, default=6145)
	# argparser.add_argument('--batch-size', type=int, default=3000)
	# argparser.add_argument('--batch-size', type=int, default=1500)
	# argparser.add_argument('--batch-size', type=int, default=8)

	argparser.add_argument("--eval-batch-size", type=int, default=100000,
                        help="evaluation batch size")
	argparser.add_argument("--R", type=int, default=5,
                        help="number of hops")
	


	argparser.add_argument('--log-every', type=int, default=5)
	argparser.add_argument('--eval-every', type=int, default=5)
	
	argparser.add_argument('--lr', type=float, default=0.003)
	argparser.add_argument('--dropout', type=float, default=0.5)
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
	
	main(args)