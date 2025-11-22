# 模仿学习训练
> 收集数据 
> python collect_data.py
> 训练bc
> python train_bc.py
> 验证效果
> python test.py
> 专家标注之前收集的训练数据, 完成DAGGER 预训练
> python dagger_train.py
> PPO 微调（核心步骤）
> python ppo_finetune.py
> 测试 PPO 微调后的 AI
> python test_ppo.py
> 20251121
> 因为模型维度改变了，之前采集的数据格式不对，需要重新执行 python collect_data.py