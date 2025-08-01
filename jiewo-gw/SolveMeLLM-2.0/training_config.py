#!/usr/bin/env python3
"""
最强模型训练配置
Training configuration for the strongest model
"""

import json
from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class TrainingConfig:
    """训练配置"""
    
    # 模型配置
    model_config = {
        'vocab_size': 50000,
        'hidden_size': 512,  # 从768减少到512
        'num_layers': 6,     # 从12减少到6
        'num_heads': 8,      # 从12减少到8
        'max_seq_length': 1024,  # 从2048减少到1024
        'dropout': 0.1,
        'activation': 'gelu',
        'layer_norm_eps': 1e-6,
        'pre_norm': True,
        'enable_clock_trigger': True,
        'clock_interval': 300,
        'enable_expression_arbitrator': True,
        'enable_cognitive_vaccine': True,
        'enable_self_iteration': True,
        'enable_active_learning': True
    }
    
    # 训练配置
    training_config = {
        'batch_size': 4,     # 从8减少到4
        'gradient_accumulation_steps': 4,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'warmup_steps': 1000,
        'max_steps': 100000,
        'max_epochs': 10,
        'jiewo_loss_weight': 0.1,
        'ethic_loss_weight': 0.2,
        'reflection_loss_weight': 0.1,
        'beta1': 0.9,
        'beta2': 0.999,
        'eps': 1e-8,
        'train_data_path': 'data/train_data.json',
        'val_data_path': 'data/val_data.json',
        'test_data_path': 'data/test_data.json',
        'save_steps': 5000,
        'eval_steps': 1000,
        'save_total_limit': 3,
        'local_rank': -1,
        'distributed': False,
        'fp16': False,
        'fp16_opt_level': 'O1',
        'logging_steps': 100,
        'log_level': 'INFO',
        'seed': 42,
        'max_grad_norm': 1.0,
        'adam_epsilon': 1e-8,
        'warmup_ratio': 0.1
    }
    
    # 数据配置
    data_config = {
        'train_file': 'complete_training_data.json',
        'eval_file': 'complete_training_data.json',  # 暂时用同一文件
        'max_seq_length': 1024,  # 与模型一致
        'overwrite_cache': False,
        'pad_to_max_length': True,
        'return_overflowing_tokens': False,
        'return_offsets_mapping': False,
        'return_length': False
    }
    
    # 安全配置
    safety_config = {
        'safety_threshold': 0.7,
        'human_reception_threshold': 0.6,
        'language_complexity_threshold': 0.8,
        'enable_safety_monitoring': True,
        'safety_check_interval': 100
    }
    
    # 优化配置
    optimization_config = {
        'use_amp': True,
        'use_apex': False,
        'fp16': True,
        'fp16_opt_level': 'O1',
        'use_distributed': False,
        'num_gpus': 1,
        'gradient_checkpointing': True,
        'save_total_limit': 3
    }
    
    # 监控配置
    monitoring_config = {
        'use_wandb': False,
        'use_tensorboard': True,
        'logging_steps': 100,
        'eval_strategy': 'steps',
        'save_strategy': 'steps',
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_loss'
    }
    
    # 解我协议训练配置
    jiewo_training_config = {
        'jiewo_state_weight': 1.0,
        'ethic_scores_weight': 0.5,
        'safety_weight': 0.3,
        'cognitive_weight': 0.2,
        'enable_jiewo_loss': True,
        'enable_ethic_loss': True,
        'enable_safety_loss': True
    }
    
    # 自我迭代配置
    iteration_config = {
        'iteration_interval': 1000,  # 每1000步执行一次自我迭代
        'max_iterations_per_training': 10,
        'improvement_threshold': 0.1,
        'confidence_threshold': 0.7,
        'enable_adaptive_iteration': True,
        'enable_self_iteration': True
    }

def create_training_config():
    """创建训练配置"""
    config = TrainingConfig()
    
    # 合并所有配置
    full_config = {
        'model': config.model_config,
        'training': config.training_config,
        'data': config.data_config,
        'safety': config.safety_config,
        'optimization': config.optimization_config,
        'monitoring': config.monitoring_config,
        'jiewo_training': config.jiewo_training_config,
        'iteration': config.iteration_config
    }
    
    # 保存配置
    with open('training_config.json', 'w', encoding='utf-8') as f:
        json.dump(full_config, f, ensure_ascii=False, indent=2)
    
    print("✅ 训练配置已保存到: training_config.json")
    
    return full_config

def print_training_summary(config: Dict[str, Any]):
    """打印训练摘要"""
    print("\n📋 训练配置摘要:")
    print("=" * 60)
    
    print("🔧 模型配置:")
    model = config['model']
    print(f"  词汇表大小: {model['vocab_size']:,}")
    print(f"  隐藏层大小: {model['hidden_size']}")
    print(f"  层数: {model['num_layers']}")
    print(f"  注意力头数: {model['num_heads']}")
    print(f"  最大序列长度: {model['max_seq_length']}")
    
    print("\n🚀 V4.0功能:")
    model = config['model']
    print(f"  Clock(τ)时序触发器: {'✅' if model['enable_clock_trigger'] else '❌'}")
    print(f"  表达裁决器: {'✅' if model['enable_expression_arbitrator'] else '❌'}")
    print(f"  认知疫苗机制: {'✅' if model['enable_cognitive_vaccine'] else '❌'}")
    print(f"  自我迭代引擎: {'✅' if model['enable_self_iteration'] else '❌'}")
    print(f"  主动学习引擎: {'✅' if model['enable_active_learning'] else '❌'}")
    
    print("\n📚 训练配置:")
    training = config['training']
    print(f"  批次大小: {training['batch_size']}")
    print(f"  学习率: {training['learning_rate']}")
    print(f"  最大步数: {training['max_steps']:,}")
    print(f"  保存间隔: {training['save_steps']}")
    print(f"  评估间隔: {training['eval_steps']}")
    
    print("\n🛡️ 安全配置:")
    safety = config['safety']
    print(f"  安全阈值: {safety['safety_threshold']}")
    print(f"  人类接收阈值: {safety['human_reception_threshold']}")
    print(f"  语言复杂度阈值: {safety['language_complexity_threshold']}")
    
    print("\n⚡ 优化配置:")
    opt = config['optimization']
    print(f"  混合精度训练: {'✅' if opt['use_amp'] else '❌'}")
    print(f"  FP16: {'✅' if opt['fp16'] else '❌'}")
    print(f"  梯度检查点: {'✅' if opt['gradient_checkpointing'] else '❌'}")
    
    print("\n🔄 自我迭代配置:")
    iteration = config['iteration']
    print(f"  迭代间隔: {iteration['iteration_interval']} 步")
    print(f"  最大迭代次数: {iteration['max_iterations_per_training']}")
    print(f"  改进阈值: {iteration['improvement_threshold']}")
    print(f"  置信度阈值: {iteration['confidence_threshold']}")
    
    print("=" * 60)

if __name__ == "__main__":
    config = create_training_config()
    print_training_summary(config) 