


class DmpALRReacherWrapper(gym.Wrapper):

	def __init__(self, env, env_start_pos):
		pass

		# self.env.start_pos
		# self.env.weight_matrix_scale  (optional)

	def step(self, a):
		pass
		# overwrite

	def reset_model(self):
		pass
		# overwrite to start at env_start_pos always (needed for stoch search)