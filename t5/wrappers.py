import gym
import numpy as np


class CmaReacherWrapper(gym.Wrapper):
    def __init__(self, env, steps_before_reward=40):

        super(CmaReacherWrapper, self).__init__(env)

        self.weight_matrix_scale = 10       # (optional?)
        self.steps_before_reward = steps_before_reward
        self.reward_weight = steps_before_reward

        self.time_limit = 1

        self.qvels = [self.start_vel]
        self.qposs = [self.start_pos]

        self._steps = 0
        # print('frame skip: ', self.env.frame_skip)

    def step(self, a):

        self._steps += 1

        reward = self._reward(a)

        self.do_simulation(a, self.frame_skip)

        done = self._steps * self.dt > self.time_limit

        self.qvels.append(self.current_vel)
        self.qposs.append(self.current_pos)

        return self._get_obs().copy(), reward, done, None

    def _reward(self, a):
        reward = 0.0
        if self._steps == self.steps_before_reward:
            reward -= np.linalg.norm(self._dist_to_goal())
            reward -= np.linalg.norm(self.current_vel)
        reward = - reward ** 2
        reward -= 1e-3 * np.square(a).sum()

        return reward

    @property
    def start_pos(self):
        return self.init_qpos[:2].copy()

    @property
    def start_vel(self):
        return self.init_qvel[:2].copy()

    @property
    def current_pos(self):
        return self.sim.data.qpos.flat[:2].copy()

    @property
    def current_vel(self):
        return self.sim.data.qvel.flat[:2].copy()

    def _dist_to_goal(self):
        return self.get_body_com("fingertip") - self.get_body_com("target")

    def reset(self):
        self.qvels = [self.start_vel]
        self.qposs = [self.start_pos]

        self._steps = 0
        qpos = self.init_qpos  # np.array([0., 0., 0.1, -0.1])
        qvel = self.init_qvel  # np.array([0., 0., 0., 0.])
        self.set_state(qpos, qvel)

        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
          np.cos(theta),
          np.sin(theta),
          self.sim.data.qpos.flat[2:],
          self.sim.data.qvel.flat[:2],
          self.get_body_com("fingertip") - self.get_body_com("target"),
          [self._steps]
        ])


