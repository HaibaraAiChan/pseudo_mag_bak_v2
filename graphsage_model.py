import dgl
import torch as th
# from torch._C import int64
import torch.nn as nn
import tqdm
import dgl.nn.pytorch as dglnn
from memory_usage import see_memory_usage

class SAGE(nn.Module):
	def __init__(self,
	             in_feats,
	             n_hidden,
	             n_classes,
	             n_layers,
	             activation,
	             dropout,
	             aggre):
		super().__init__()
		self.n_layers = n_layers
		self.n_hidden = n_hidden
		self.n_classes = n_classes
		self.aggre = aggre
		self.layers = nn.ModuleList()
		if n_layers == 1:
			self.layers.append(dglnn.SAGEConv(in_feats, n_classes, self.aggre))
		if n_layers >= 2:
			self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, self.aggre))
			for i in range(0, n_layers - 2):
				self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, self.aggre))
			self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, self.aggre))
		self.dropout = nn.Dropout(dropout)

		self.activation = activation

	# def forward(self, blocks, x):
	# 	h = x
	# 	for l, (layer, block) in enumerate(zip(self.layers, blocks)):
	# 		h = layer(block, h)
	# 		if l!=len(self.layers) - 1:
	# 			h = self.activation(h)
	# 			h = self.dropout(h)
	# 	return h

	def forward(self, blocks, x):
		h = x
		# print('x')
		# print(x)
		for l, (layer, block) in enumerate(zip(self.layers, blocks)):
			# print('block')
			# print(block.ndata['_ID'])
			# print(block.srcdata)
			# print(block.dstdata)
			# print(block)
			# see_memory_usage('before layer (block, h)')
			h = layer(block, h)
			# see_memory_usage('after layer (block, h)')
			# print('h')
			# print(h)
			if l!=len(self.layers) - 1:
				h = self.activation(h)
				h = self.dropout(h)

		return h

	def inference(self, g, x, device, args):
		"""
		Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
		g : the entire graph.
		x : the input of entire node set.

		The inference code is written in a fashion that it could handle any number of nodes and
		layers.
		"""
		# During inference with sampling, multi-layer blocks are very inefficient because
		# lots of computations in the first few layers are repeated.
		# Therefore, we compute the representation of all nodes layer by layer.  The nodes
		# on each layer are of course splitted in batches.
		# TODO: can we standardize this?
		for l, layer in enumerate(self.layers):
			y = th.zeros(g.num_nodes(), self.n_hidden if l!=len(self.layers) - 1 else self.n_classes)

			sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
			dataloader = dgl.dataloading.NodeDataLoader(
				g,
				th.arange(g.num_nodes(),dtype=th.long),
				sampler,
				batch_size=args.batch_size,
				shuffle=True,
				drop_last=False,
				num_workers=args.num_workers)

			# print('x')
			# print(type(x))
			# print(len(x))
			# print(x[0].size())

			for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
				block = blocks[0]
				# print('input_nodes')
				# print(type(input_nodes))
				# input_nodes=input_nodes.tolist()
				# print(type(input_nodes))
				# print(input_nodes)


				block = block.int().to(device)
				h = x[input_nodes].to(device)
				h = layer(block, h)
				if l!=len(self.layers) - 1:
					h = self.activation(h)
					h = self.dropout(h)

				y[output_nodes] = h.cpu()

			x = y
		return y

