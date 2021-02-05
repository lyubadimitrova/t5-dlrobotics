import gym
import numpy as np

class CmaReacherWrapper(gym.Wrapper):

    def __init__(self, env, params={}):

        super(CmaReacherWrapper, self).__init__(env)

        self.weight_matrix_scale = 10       # (optional?)

        self.time_limit = 1

        self.start_pos = self.init_qpos[:2]
        self.start_vel = self.init_qvel[:2]

        self.qvels = [self.init_qvel]
        self.qposs = [self.init_qpos]
        self.xvelps = []
        self.xvelrs = []
        self.actions_taken = []

        self._steps = 0
        # print('frame skip: ', self.env.frame_skip)

    def step(self, vel):
        
        # self.actions_taken.append(vel)
        # print(vel)
        # print('jv ', self.joint_vels)
        old_vel = self.joint_vels
        acc = (vel - old_vel) / self.dt

        self.xvelrs.append(old_vel)
        reward = self._reward(vel, acc)
        self._steps += 1
        self.do_simulation(acc, self.frame_skip) 
        done = self._steps * self.dt > self.time_limit

        self.qvels.append(self.sim.data.qvel.flat[:])
        self.qposs.append(self.sim.data.qpos.flat[:])

        # print(self._steps)
        return self._get_obs().copy(), reward, done, None

    def _reward(self, vel, acc):
        reward = 0

        if self._steps == 40:  # max episode length for Reacher-v2 is 50
            reward = np.linalg.norm(self._dist_to_goal())  # dist reward

        reward = - reward ** 2
        # reward -= np.square(vel).sum()
        # reward -= 1e-6 * np.sum(acc**2)  # penalize high accelerations
        reward -= 1e-6 * np.square(acc).sum()
        if self._steps == 40:
            reward -= 0.001 * np.sum(vel**2) ** 2  # fingertip not moving too fast

        return reward

    @property
    def _qvel(self):
        return self.sim.data.qvel.flat[:2]

    @property
    def joint_vels(self):
        joint0 = self.sim.data.get_body_xvelr('body0')[-1]  # z-axis rotational velocity
        joint1 = self.sim.data.get_body_xvelr('body1')[-1]
        return np.array([joint0, joint1])

    def _dist_to_goal(self):
        return self.get_body_com("fingertip") - self.get_body_com("target")

    def reset(self):
        self.qvels = [self.init_qvel]
        self.qposs = [self.init_qpos]
        self.actions_taken = []
        self.xvelrs = []
        self.xvelps = []

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
          # self.get_body_com("target"),
          self.sim.data.qpos.flat[2:],
          self.sim.data.qvel.flat[:2],
          self.get_body_com("fingertip") - self.get_body_com("target"),
          [self._steps]
        ])

