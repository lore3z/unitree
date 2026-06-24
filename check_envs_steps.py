import sys
import os
sys.path.append(os.getcwd())
from legged_gym.envs.g1.g1_config import G1Cfg, G1CfgPPO
from legged_gym.envs.g1.g1_arms_config import G1ArmsCfg, G1ArmsCfgPPO
print("G1 (12 DoF):", G1Cfg.env.num_envs, G1CfgPPO.runner.num_steps_per_env, G1CfgPPO.runner.max_iterations)
print("G1 Arms (29 DoF):", G1ArmsCfg.env.num_envs, G1ArmsCfgPPO.runner.num_steps_per_env, G1ArmsCfgPPO.runner.max_iterations)
