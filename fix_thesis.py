import re

with open('thesis/main.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Fix abstract
text = re.sub(
    r'相比于传统的 12-DoF 下肢基线，本文重点引入了包含手臂与腰部的 29-DoF 全身模型，并在奖励函数设计中充分考虑了指令跟踪、能耗约束、动作平滑和姿态修正等多重目标。实验观察到，尽管高维状态空间在训练初期增加了探索难度，但策略最终学会了利用摆臂动量来补偿足端冲击产生的偏航误差，显著提升了机器人在复杂指令下的动态稳定性。',
    r'本文在基线基础之上，引入了包含手臂与腰部的 29-DoF 全身动力学模型进行仿真，但在策略网络端采取了控制降维方案：策略仅输出 12 个腿部关节的动作，而上肢等额外自由度则依靠底层 PD 控制器维持在预设角度。这种具有实际物理约束的环境迫使策略在面临复杂的上半身被动惯量与受力扰动时，学习更加鲁棒的下肢步态。在奖励函数设计中，本文充分考虑了速度跟踪、躯干高度维持、脚摆高度约束以及动作平滑等多重目标。实验观察到，在 29-DoF 全身重力耦合下获得的下肢策略，显著提升了机器人在复杂指令下的动态稳定性。',
    text
)

# 2. Fix 1.2 "17de59a" parts
text = re.sub(
    r'最早在 17de59a 把 G1 任务骨架搭起来时，主要目标还是先让机器人学会“能走”；到了 bd00c6a 对 g1、h1、h1_2 的训练逻辑做统一调整后，动作平滑、关节限幅和姿态约束才真正开始压住那些看起来能拿分、但实际上不太能落地的动作；再往后做 push robot 相关修改时',
    r'在项目初期搭建基础任务骨架时，主要目标是先让机器人学会“能走”；而在后续对不同机器人平台的训练逻辑进行统一重构后，动作平滑、关节限幅和姿态约束等项才真正开始发挥核心作用，抑制了那些虽然能获取高分但难以在现实中落地的动作；再往后引入机器人受力扰动（Push Robot）的训练模块时',
    text
)

text = re.sub(
    r'最早的 17de59a 先把 G1 任务和基础配置搭起来，重点是让训练链路先跑通；到了 bd00c6a，g1、h1_2 的训练逻辑开始一起调整',
    r'在项目早期的原型版本中，首先搭建了 G1 的任务及基础配置，重点是为了打通训练链路；随后进入架构重构阶段，对多平台的训练逻辑进行了统一调整',
    text
)

text = re.sub(
    r'从版本迭代上看，代码历史之所以有意义，是因为它记录了这些问题是怎么一步步被发现的。17de59a 时先搭任务骨架，说明当时最紧要的是把环境和训练链路接上；bd00c6a 开始统一 g1、h1_2 的训练逻辑，说明那一阶段已经在做跨任务的参数收敛；到了后面和 push robot 相关的修改阶段',
    r'从迭代历程上看，工程记录展示了问题逐步暴露与解决的脉络。在初步构建阶段搭建任务骨架，重点在于走通仿真与训练的闭环流程；在架构统一阶段整合多个平台的训练基类，表明研究焦点已转向跨环境的参数一致性收敛；而到了后期加入外部扰动（如施加推力）的阶段',
    text
)

# 3. Fix state dimensions and observations in 3.1
text = re.sub(
    r's_t = \\left\[ g, v_t\^{cmd}, \\omega_t\^{cmd}, q_t, \\dot{q}_t, a_{t-1} \\right\] \\in \\mathbb{R}\^{d_{obs}}',
    r's_t = \\left[ \\omega_t^{base}, g, v_t^{cmd}, \\Delta q_t, \\dot{q}_t, a_{t-1}, \\text{phases} \\right] \\in \\mathbb{R}^{81}',
    text
)

text = re.sub(
    r'\\begin{itemize}.*?其中状态向量的维度 d_{obs} 对于 12-DoF 模型为.*?d_{act} 分别为 12 和 29。',
    r'''\\begin{itemize}
    \\item \\textbf{基座角速度 $\\omega_t^{base}$} (3维)：实时感知的机体滚转、俯仰、偏航角速度，用于动态平衡调节；
    \\item \\textbf{重力向量 $g \\in \\mathbb{R}^3$}：通过 IMU 实时获取躯干在世界坐标系中的重力方向投影，为姿态控制提供极低延迟的基准参考；
    \\item \\textbf{速度指令 $v_t^{cmd} \\in \\mathbb{R}^3$}：包含目标线速度和转向要求；
    \\item \\textbf{关节位置偏差 $\\Delta q_t \\in \\mathbb{R}^{29}$}：当前29个关节的实际位置与默认站立位置的差值；
    \\item \\textbf{关节速度 $\\dot{q}_t \\in \\mathbb{R}^{29}$}：所有29个关节的作用角速度；
    \\item \\textbf{先序动作 $a_{t-1} \\in \\mathbb{R}^{12}$}：前一时步策略网络下发的下肢12关节动作；
    \\item \\textbf{步态相位 $\\text{phases} \\in \\mathbb{R}^2$}：基于正弦和余弦函数编码的参考周期信号，辅助机器人建立具有节律的迈步能力。
\\end{itemize}

综上所述，当前输入状态向量的维度固定在 81 维，而网络在输出端（动作空间）维度被严格限定为 12 维，即仅实时控制下盘的 12 个驱动关节。''',
    text,
    flags=re.DOTALL
)

# 4. Update the Reward explanation and table
text = re.sub(
    r'\\begin{table}\[h\]\n\\centering\n\\caption{多维度复合奖励函数关键构成与权重配比}\n\\begin{tabular}{llcc}.*?\\bottomrule\n\\end{tabular}\n\\end{table}',
    r'''\\begin{table}[h]
\\centering
\\caption{核心奖励函数设计与权重配比}
\\begin{tabular}{llcc}
\\toprule
\\textbf{奖励类别} & \\textbf{设计意图} & \\textbf{权重 w} & \\textbf{物理涵义} \\\\
\\midrule
\\multirow{2}{*}{速度跟踪} & 线速度跟踪 & +2.0 & 促使机器人前行或侧移符合指令 \\\\
         & 角速度跟踪 & +0.5 & 保障对转向指令的快速响应 \\\\
\\midrule
\\multirow{1}{*}{姿态与轨迹约束} & 维持基础身高 & -10.0 & 惩罚质心过低或下蹲行走 \\\\
         & 约束脚摆高度 & -20.0 & 规范跨步时的抬脚离地轨迹 \\\\
\\midrule
\\multirow{1}{*}{能耗与平滑} & 关节加速度平滑 & -2.5e-7 & 抑制运动中出现的异常加速度抖动 \\\\
\\bottomrule
\\end{tabular}
\\end{table}''',
    text,
    flags=re.DOTALL
)

# 5. Fix environment counts 4096 -> 1024
text = text.replace('4096 个并行环境', '1024 个并行环境')
text = text.replace('4096 倍的并行化', '1024 倍的并行化')
text = text.replace('放大了 4096 倍', '放大了 1024 倍')

with open('thesis/main.tex', 'w', encoding='utf-8') as f:
    f.write(text)
