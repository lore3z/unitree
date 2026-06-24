from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
           'left_hip_yaw_joint' : 0. ,   
           'left_hip_roll_joint' : 0,               
           'left_hip_pitch_joint' : -0.1,         
           'left_knee_joint' : 0.3,       
           'left_ankle_pitch_joint' : -0.2,     
           'left_ankle_roll_joint' : 0,     
           'right_hip_yaw_joint' : 0., 
           'right_hip_roll_joint' : 0, 
           'right_hip_pitch_joint' : -0.1,                                       
           'right_knee_joint' : 0.3,                                             
           'right_ankle_pitch_joint': -0.2,                              
           'right_ankle_roll_joint' : 0,       
           'torso_joint' : 0.
        }
    
    class env(LeggedRobotCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 2.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1'


class G1ArmsCfg(G1RoughCfg):
    """G1 with full 29-DOF model, policy still controls 12 leg joints.

    - 使用完整 g1_29dof_rev_1_0.urdf，把上肢质量和碰撞都纳入；
    - 动作维度保持 12，不改 deploy 侧接口；
    - 手臂/腰部通过 PD 固定在默认角度，可被动受力（"解放" 结构但不直接由策略驱动）。
    """

    class init_state(G1RoughCfg.init_state):
        pos = [0.0, 0.0, 0.8] # Increase height slightly back to 0.8
        # 补全所有关节的默认角度，确保初始化姿态正常
        default_joint_angles = {
            'left_hip_yaw_joint': 0.0,
            'left_hip_roll_joint': 0.0,
            'left_hip_pitch_joint': -0.1,
            'left_knee_joint': 0.3,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_hip_roll_joint': 0.0,
            'right_hip_pitch_joint': -0.1,
            'right_knee_joint': 0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,
            'waist_yaw_joint': 0.0,
            'waist_roll_joint': 0.0,
            'waist_pitch_joint': 0.0,
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 1.3,
            'left_wrist_roll_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': 0.0,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 1.3,
            'right_wrist_roll_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
        }

    class env(G1RoughCfg.env):
        num_envs = 1024 # Reduce number of envs to fit in GPU memory
        # 仍然只控制 12 个腿 DOF，和原 g1 对齐
        num_actions = 12
        # 观测结构沿用 G1Robot：3(ang_vel)+3(gravity)+3(cmd)+2*num_dof(pos/vel)+num_actions(act)+2(phase)
        # 3+3+3+29+29+12+2 = 81
        num_observations = 81
        num_privileged_obs = 84

    class control(G1RoughCfg.control):
        control_type = 'P'
        stiffness = {
            'hip_yaw': 80,
            'hip_roll': 80,
            'hip_pitch': 80,
            'knee': 100,
            'ankle': 30,
            'waist': 100,    # 核心修改：大幅提高腰部刚度（20->100），锁定上半身防止倾斜
            'shoulder_pitch': 2, # 核心修改：大幅降低肩部俯仰刚度（5->2），利用自然惯性实现被动摆臂
            'shoulder_roll': 10, # 提高防止手臂外展
            'shoulder_yaw': 10, 
            'elbow': 20,     # 提高肘部刚度，保持手臂形态
            'wrist': 2,
        }
        damping = {
            'hip_yaw': 2,
            'hip_roll': 2,
            'hip_pitch': 2,
            'knee': 3,
            'ankle': 1.5,
            'waist': 4,      # 增加腰部阻尼，消除躯干晃动
            'shoulder_pitch': 0.5, # 降低肩部阻尼，让手臂更顺滑地摆动
            'shoulder_roll': 1,
            'shoulder_yaw': 1,
            'elbow': 1,
            'wrist': 0.1,
        }
        action_scale = 0.25 
        decimation = 10

    class domain_rand(G1RoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5

    class sim(G1RoughCfg.sim):
        dt = 0.001
        substeps = 1
        class physx(G1RoughCfg.sim.physx):
            num_position_iterations = 8
            num_velocity_iterations = 1

    class asset(G1RoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof_rev_1_0.urdf'
        name = 'g1_arms'
        foot_name = 'ankle_roll'
        penalize_contacts_on = ['hip', 'knee']
        terminate_after_contacts_on = ['pelvis']
        self_collisions = 1 # Disable self collisions
        flip_visual_attachments = False
        fix_base_link = False
        armature = 0.06
        replace_cylinder_with_capsule = False
        collapse_fixed_joints = False

    class rewards(G1RoughCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78 # G1RoughCfg uses LeggedRobotCfg default which is 1.0 but overridden in G1RoughCfg to 0.78? Let's check. 
        # G1RoughCfg defines rewards class and overrides things.
        # But wait, looking at G1RoughCfg above, it defines base_height_target = 0.78
        # So I don't need to redefine it unless I want to change it.
        # But I need to be careful about what G1RoughCfg inheritance gives me.
        pass # Inherit rewards from G1RoughCfg directly to be safe.



class G1ArmsCfgPPO(G1RoughCfgPPO):
    class policy(G1RoughCfgPPO.policy):
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu'

    class algorithm(G1RoughCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(G1RoughCfgPPO.runner):
        policy_class_name = 'ActorCriticRecurrent'
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1_arms'

  
