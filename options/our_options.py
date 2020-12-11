from .test_options import TestOptions


class OurOptions(TestOptions):
	def initialize(self):
		TestOptions.initialize(self)
		self.parser.add_argument('--embedding_size', type=int, default=64)
		self.parser.add_argument('--num_heads', type=int, default=2)
		self.parser.add_argument('--window_size', type=int, default=25)
