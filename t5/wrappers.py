import gym
import numpy as np

class CmaReacherWrapper(gym.Wrapper):

    def __init__(self, env, params={}):

        super(CmaReacherWrapper, self).__init__(env)

        self.env = env
        #print(self.env.state_vector())
        #self.start_pos = self.env.init_qpos
        self.start_pos = np.zeros(2) #np.array([0, 0, 0.1, -0.1])
        self.start_vel = np.zeros(2)
        self.weight_matrix_scale = 50        # (optional?)

        self._dt = 0.01
        self.time_limit = 2

    def step(self, vel):
        acc = (vel - self._qvel) / self._dt   # use mujoco env.dt?
        reward = self._reward(vel, acc)
        self._steps += 1
        self.env.do_simulation(vel, self.env.frame_skip)
        done = self._steps > 50

        return self._get_obs().copy(), reward, done, None

    def _reward(self, vel, acc):
        reward = 0

        if self._steps == 40:  # max episode length for Reacher-v2 is 50
            reward += -np.linalg.norm(self._dist_to_goal())  # dist reward

        reward = - reward ** 2
        reward -= 1e-6 * np.sum(acc**2)  # penalize high accelerations?

        if self._steps == 40:
            reward -= 0.1 * np.sum(vel**2) ** 2

        return reward

    @property
    def _qvel(self):
        return self.env.sim.data.qvel.flat[:2]

    def _dist_to_goal(self):
        return self.env.get_body_com("fingertip") - self.env.get_body_com("target")

    def reset(self):
        self._steps = 0
        qpos = np.array([-0.00824902, 0.03426524, 0.1, -0.1])
        qvel = np.array([-0.8221599, 3.41514151, 0., 0.])
        self.set_state(qpos, qvel)

        return self._get_obs()
        # overwrite to start at env_start_pos always (needed for stoch search)

    def _get_obs(self):
        #print(self.env)
        #return np.concatenate([self.env._get_obs(), self._steps])

        theta = self.env.sim.data.qpos.flat[:2]
        return np.concatenate([
          np.cos(theta),
          np.sin(theta),
          # self.get_body_com("target"),
          self.env.sim.data.qpos.flat[2:],
          self.env.sim.data.qvel.flat[:2],
          self.env.get_body_com("fingertip") - self.env.get_body_com("target"),
          [self._steps]
        ])

