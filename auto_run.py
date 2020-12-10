import os
import subprocess


class HyperParameters:
	def __init__(self, type='classification', batch_size=16, resblocks=1, fc_n=100, ncf=[16, 32, 32], pool_res=[1140, 780, 580],
				 init_type='normal', init_gain=0.02, lr=0.0002, beta1=0.9, continue_train=True, embedding_size=64,
				 num_heads=2, window_size=25):
		self.type = type
		self.batch_size = batch_size
		self.resblocks = resblocks
		self.fc_n = fc_n
		self.ncf = ncf
		self.pool_res = pool_res
		self.init_type = init_type
		self.init_gain = init_gain
		self.lr = lr
		self.beta1 = beta1
		self.continue_train = continue_train
		self.embedding_size = embedding_size
		self.num_heads = num_heads
		self.window_size = window_size
		self.dataset = 'cubes' if self.type == 'classification' else 'humans'

	def __str__(self):
		return f"python train.py --dataroot datasets/{self.dataset}" \
		f"--name {self.dataset}" \
		f"--batch_size {self.batch_size} --resblocks {self.resblocks} --fc_n {self.fc_n}" \
		f"--ncf {''.join(f'{i} ' for i in self.ncf)}" \
		f"--pool_res {''.join(f'{i} ' for i in self.pool_res)} --init_type {self.init_type} --init_gain {self.init_gain}" \
		f"--lr {self.lr}" f"--beta1 {self.beta1} {'--continue_train' if self.continue_train else ''}" \
		f"--embedding_size {self.embedding_size}" \
		f"--num_heads {self.num_heads} --window_size {self.window_size}"

	def to_list(self):
		return [
			f"--dataroot datasets/{self.dataset}",
			f"--name {self.dataset}",
			f"--batch_size {self.batch_size}",
			f"--resblocks {self.resblocks}",
			f"--fc_n {self.fc_n}",
			f"--ncf {''.join(f'{i} ' for i in self.ncf)}",
			f"--pool_res {''.join(f'{i} ' for i in self.pool_res)}",
			f"--init_type {self.init_type}",
			f"--init_gain {self.init_gain}",
			f"--lr {self.lr}",
			f"--beta1 {self.beta1}",
			f"{'--continue_train' if self.continue_train else ''}",
			f"--embedding_size {self.embedding_size}",
			f" - -num_heads {self.num_heads}",
			f"--window_size {self.window_size}"
		]


hyper_parameters = [
	HyperParameters(),
	HyperParameters(batch_size=32, resblocks=10, fc_n=120, ncf=[16, 32, 64], pool_res=[1200, 790, 600], lr=0.002)
]

for hyper_paramter in hyper_parameters:
	params = hyper_paramter.to_list()
	subprocess.run(["python", "train.py"].extend(params))
