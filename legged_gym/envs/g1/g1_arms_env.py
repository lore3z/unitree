from legged_gym.envs.g1.g1_env import G1Robot


class G1ArmsRobot(G1Robot):
    """Env class for g1_arms.

    继承 G1Robot：
    - 观测/奖励结构完全复用 g1；
    - 区别只在于 config 里选用 29-DOF URDF 和不同的 PD 刚度。
    这样可以确保 g1 的行为不受影响，g1_arms 单独训练/测试。
    """

    # 暂时不需要额外 override，如之后要给手臂加专门 reward，可以在这里扩展。
    pass

    def compute_observations(self):
        """ Computes observations for 29 dof arm model (Dim 81)
        """
        import torch
        sin_phase = torch.sin(2 * torch.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * torch.pi * self.phase).unsqueeze(1)
        
        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            # NOT including last actions for some reason to match 81?
            # base_ang_vel(3) + gravity(3) + cmd(3) + dof_pos(29) + dof_vel(29) + actions(29)? Wait, 3+3+3+29+29+29 = 96.
            # Maybe the original custom model is just 12 DOF actions (12) + dof_pos(29) + dof_vel(29) + 9 = 79 + 2 phase = 81!
            self.actions[:, :12], # Only leg actions!
            sin_phase,
            cos_phase
        ), dim=-1)

        self.privileged_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.obs_buf
        ), dim=-1)
