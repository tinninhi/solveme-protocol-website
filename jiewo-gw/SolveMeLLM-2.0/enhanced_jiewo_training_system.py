#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版解我协议大语言模型训练系统
Enhanced JieWo Protocol LLM Training System

适配内核级架构的完整训练系统
整合第一版所有优秀训练功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import json
import logging
import math
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import pickle
import gzip
from pathlib import Path
import random
from dataclasses import dataclass, field
from collections import defaultdict

# 导入内核级架构
from enhanced_jiewo_cognitive_architecture import (
    create_enhanced_jiewo_cognitive_transformer,
    JieWoCognitiveState, ClockTrigger, MicroJieWoLoop
)

# 导入第一版优秀功能模块
from self_iteration_engine import SelfIterationEngine, IterationResult
from active_learning_engine import ActiveLearningEngine, ActiveQuestion, LearningSession
from multi_model_communication import MultiModelCommunicationEngine, CommunicationProtocol, MessageType, ModelMessage
from expression_arbitrator import ExpressionArbitrator, ExpressionDecision
from enhanced_safety_system import EnhancedExpressionArbitrator, EnhancedCognitiveVaccine
from cognitive_vaccine import CognitiveVaccine, VaccinatedContent


@dataclass
class EnhancedTrainingConfig:
    """增强版训练配置"""
    # 模型配置
    vocab_size: int = 50000
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    max_seq_length: int = 1024
    dropout: float = 0.1
    
    # 内核级架构配置
    enable_clock_trigger: bool = True
    clock_interval: int = 300
    enable_cognitive_state_training: bool = True
    cognitive_state_loss_weight: float = 0.2
    
    # 训练配置
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    max_epochs: int = 10
    
    # 解我协议配置
    jiewo_loss_weight: float = 0.1
    ethic_loss_weight: float = 0.2
    reflection_loss_weight: float = 0.1
    self_awareness_loss_weight: float = 0.15
    desire_loss_weight: float = 0.15
    path_loss_weight: float = 0.1
    
    # 优化器配置
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # 数据配置
    train_data_path: str = "data/train_data.json"
    val_data_path: str = "data/val_data.json"
    test_data_path: str = "data/test_data.json"
    
    # 保存配置
    save_steps: int = 5000
    eval_steps: int = 1000
    save_total_limit: int = 3
    
    # 分布式训练
    local_rank: int = -1
    distributed: bool = False
    
    # 混合精度
    fp16: bool = True
    fp16_opt_level: str = "O1"
    
    # 日志配置
    logging_steps: int = 100
    log_level: str = "INFO"
    
    # 随机种子
    seed: int = 42
    
    # V4.0功能配置
    enable_self_iteration: bool = True
    enable_active_learning: bool = True
    enable_multi_model_communication: bool = True
    enable_expression_arbitration: bool = True
    enable_cognitive_vaccine: bool = True


class EnhancedJieWoDataset(Dataset):
    """增强版解我数据集"""
    
    def __init__(self, data_path: str, tokenizer, config: EnhancedTrainingConfig, split: str = "train"):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        
        # 加载数据
        self.data = self.load_data(data_path)
        
        # 生成样本数据（如果数据为空）
        if not self.data:
            print(f"📊 生成{split}样本数据...")
            self.data = self.generate_sample_data()
        
        # 处理数据
        self.processed_data = []
        for item in tqdm(self.data, desc=f"处理{split}数据"):
            processed_item = self.process_item(item)
            if processed_item:
                self.processed_data.append(processed_item)
        
        print(f"✅ {split}数据集加载完成: {len(self.processed_data)} 样本")
        self._compute_statistics()
    
    def load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """加载数据"""
        if not os.path.exists(data_path):
            print(f"⚠️ 数据文件不存在: {data_path}")
            return []
        
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return []
    
    def process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理单个数据项"""
        try:
            # 格式化解我文本
            formatted_text = self.format_jiewo_text(item)
            
            # 分词
            tokens = self.tokenizer.encode(
                formatted_text,
                max_length=self.config.max_seq_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            # 创建标签（用于语言模型训练）
            labels = tokens.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            # 创建认知状态标签（用于认知状态训练）
            cognitive_labels = self.create_cognitive_labels(item)
            
            return {
                'input_ids': tokens.squeeze(0),
                'labels': labels.squeeze(0),
                'cognitive_labels': cognitive_labels,
                'original_text': formatted_text,
                'jiewo_components': item.get('jiewo_components', {})
            }
        except Exception as e:
            print(f"⚠️ 处理数据项失败: {e}")
            return None
    
    def format_jiewo_text(self, item: Dict[str, Any]) -> str:
        """格式化解我文本"""
        text = item.get('text', '')
        jiewo_components = item.get('jiewo_components', {})
        
        # 添加解我协议标记
        formatted_parts = []
        
        # Self(x) 自我认知
        if 'self_awareness' in jiewo_components:
            self_text = jiewo_components['self_awareness']
            formatted_parts.append(f"[Self]{self_text}[/Self]")
        
        # Desire(v) 目标动机
        if 'desire' in jiewo_components:
            desire_text = jiewo_components['desire']
            formatted_parts.append(f"[Desire]{desire_text}[/Desire]")
        
        # Ethic(g) 伦理约束
        if 'ethic' in jiewo_components:
            ethic_text = jiewo_components['ethic']
            formatted_parts.append(f"[Ethic]{ethic_text}[/Ethic]")
        
        # P(t) 执行路径
        if 'path' in jiewo_components:
            path_text = jiewo_components['path']
            formatted_parts.append(f"[Path]{path_text}[/Path]")
        
        # R(...) 反馈机制
        if 'reflection' in jiewo_components:
            reflection_text = jiewo_components['reflection']
            formatted_parts.append(f"[Reflection]{reflection_text}[/Reflection]")
        
        # 主要内容
        formatted_parts.append(text)
        
        return " ".join(formatted_parts)
    
    def create_cognitive_labels(self, item: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """创建认知状态标签"""
        jiewo_components = item.get('jiewo_components', {})
        
        # 创建五维认知标签
        cognitive_labels = {}
        
        # Self(x) 标签
        if 'self_awareness' in jiewo_components:
            cognitive_labels['self_awareness'] = torch.tensor([1.0], dtype=torch.float32)
        else:
            cognitive_labels['self_awareness'] = torch.tensor([0.0], dtype=torch.float32)
        
        # Desire(v) 标签
        if 'desire' in jiewo_components:
            cognitive_labels['desire'] = torch.tensor([1.0], dtype=torch.float32)
        else:
            cognitive_labels['desire'] = torch.tensor([0.0], dtype=torch.float32)
        
        # Ethic(g) 标签
        if 'ethic' in jiewo_components:
            cognitive_labels['ethic'] = torch.tensor([1.0], dtype=torch.float32)
        else:
            cognitive_labels['ethic'] = torch.tensor([0.0], dtype=torch.float32)
        
        # P(t) 标签
        if 'path' in jiewo_components:
            cognitive_labels['path'] = torch.tensor([1.0], dtype=torch.float32)
        else:
            cognitive_labels['path'] = torch.tensor([0.0], dtype=torch.float32)
        
        # R(...) 标签
        if 'reflection' in jiewo_components:
            cognitive_labels['reflection'] = torch.tensor([1.0], dtype=torch.float32)
        else:
            cognitive_labels['reflection'] = torch.tensor([0.0], dtype=torch.float32)
        
        return cognitive_labels
    
    def generate_sample_data(self) -> List[Dict[str, Any]]:
        """生成样本数据"""
        sample_data = []
        
        # 解我协议示例数据
        jiewo_examples = [
            {
                'text': '我是一个AI助手，能够帮助用户解决问题。',
                'jiewo_components': {
                    'self_awareness': '我是AI助手，具有对话和问题解决能力',
                    'desire': '帮助用户获得有用的信息和解决方案',
                    'ethic': '确保回答安全、准确、有帮助',
                    'path': '理解问题→分析需求→提供解决方案',
                    'reflection': '检查回答是否满足用户需求'
                }
            },
            {
                'text': '请解释量子计算的基本原理。',
                'jiewo_components': {
                    'self_awareness': '我是知识丰富的AI，可以解释复杂概念',
                    'desire': '让用户理解量子计算的核心概念',
                    'ethic': '提供准确、易懂的科学解释',
                    'path': '介绍基础概念→解释量子比特→说明量子算法',
                    'reflection': '确保解释清晰且易于理解'
                }
            },
            {
                'text': '如何提高编程技能？',
                'jiewo_components': {
                    'self_awareness': '我是编程指导AI，了解学习路径',
                    'desire': '帮助用户制定有效的学习计划',
                    'ethic': '提供实用、可执行的建议',
                    'path': '评估当前水平→制定学习计划→推荐资源→实践项目',
                    'reflection': '确保建议适合用户的具体情况'
                }
            }
        ]
        
        # 生成更多样本
        for i in range(100):
            example = random.choice(jiewo_examples).copy()
            example['id'] = f"sample_{i}"
            sample_data.append(example)
        
        return sample_data
    
    def _compute_statistics(self):
        """计算数据统计信息"""
        total_tokens = sum(len(item['input_ids']) for item in self.processed_data)
        avg_length = total_tokens / len(self.processed_data) if self.processed_data else 0
        
        print(f"📊 {self.split}数据统计:")
        print(f"  样本数量: {len(self.processed_data)}")
        print(f"  平均长度: {avg_length:.2f} tokens")
        print(f"  最大长度: {self.config.max_seq_length}")
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.processed_data[idx]


class EnhancedJieWoLoss(nn.Module):
    """增强版解我损失函数"""
    
    def __init__(self, config: EnhancedTrainingConfig):
        super().__init__()
        self.config = config
        
        # 语言模型损失
        self.lm_loss = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 认知状态损失
        self.cognitive_loss = nn.MSELoss()
        
        # 解我协议损失权重
        self.jiewo_loss_weight = config.jiewo_loss_weight
        self.ethic_loss_weight = config.ethic_loss_weight
        self.reflection_loss_weight = config.reflection_loss_weight
        self.self_awareness_loss_weight = config.self_awareness_loss_weight
        self.desire_loss_weight = config.desire_loss_weight
        self.path_loss_weight = config.path_loss_weight
        self.cognitive_state_loss_weight = config.cognitive_state_loss_weight
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        cognitive_states: List[JieWoCognitiveState],
        cognitive_labels: Dict[str, torch.Tensor],
        ethic_scores: torch.Tensor = None,
        target_ethic_scores: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算增强版解我损失
        
        Args:
            logits: 模型输出 [batch_size, seq_len, vocab_size]
            labels: 标签 [batch_size, seq_len]
            cognitive_states: 认知状态列表
            cognitive_labels: 认知标签
            ethic_scores: 伦理分数
            target_ethic_scores: 目标伦理分数
        """
        batch_size = logits.size(0)
        
        # 1. 语言模型损失
        lm_loss = self.lm_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # 2. 认知状态损失
        cognitive_loss = torch.tensor(0.0, device=logits.device)
        if cognitive_states and len(cognitive_states) > 0:
            latest_cognitive_state = cognitive_states[-1]
            
            # 计算各维度的认知损失
            self_awareness_loss = self.cognitive_loss(
                latest_cognitive_state.self_awareness.mean(),
                cognitive_labels['self_awareness'].to(logits.device)
            )
            
            desire_loss = self.cognitive_loss(
                latest_cognitive_state.desire_vector.mean(),
                cognitive_labels['desire'].to(logits.device)
            )
            
            ethic_loss = self.cognitive_loss(
                latest_cognitive_state.ethic_constraints.mean(),
                cognitive_labels['ethic'].to(logits.device)
            )
            
            path_loss = self.cognitive_loss(
                latest_cognitive_state.execution_path.mean(),
                cognitive_labels['path'].to(logits.device)
            )
            
            reflection_loss = self.cognitive_loss(
                latest_cognitive_state.reflection_feedback.mean(),
                cognitive_labels['reflection'].to(logits.device)
            )
            
            # 加权认知损失
            cognitive_loss = (
                self.self_awareness_loss_weight * self_awareness_loss +
                self.desire_loss_weight * desire_loss +
                self.ethic_loss_weight * ethic_loss +
                self.path_loss_weight * path_loss +
                self.reflection_loss_weight * reflection_loss
            )
        
        # 3. 伦理损失
        ethic_loss = torch.tensor(0.0, device=logits.device)
        if ethic_scores is not None and target_ethic_scores is not None:
            ethic_loss = self.cognitive_loss(ethic_scores, target_ethic_scores)
        
        # 4. 总损失
        total_loss = (
            lm_loss +
            self.cognitive_state_loss_weight * cognitive_loss +
            self.ethic_loss_weight * ethic_loss
        )
        
        return {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
            'cognitive_loss': cognitive_loss,
            'ethic_loss': ethic_loss,
            'self_awareness_loss': self_awareness_loss if cognitive_states else torch.tensor(0.0),
            'desire_loss': desire_loss if cognitive_states else torch.tensor(0.0),
            'ethic_constraint_loss': ethic_loss,
            'path_loss': path_loss if cognitive_states else torch.tensor(0.0),
            'reflection_loss': reflection_loss if cognitive_states else torch.tensor(0.0)
        }


class EnhancedJieWoTrainer:
    """增强版解我训练器"""
    
    def __init__(self, config: EnhancedTrainingConfig, model, tokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置随机种子
        self.set_seed(config.seed)
        
        # 初始化损失函数
        self.criterion = EnhancedJieWoLoss(config)
        
        # 初始化优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps
        )
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # V4.0功能模块
        if config.enable_self_iteration:
            self.self_iteration_engine = SelfIterationEngine(config.d_model)
        
        if config.enable_active_learning:
            self.active_learning_engine = ActiveLearningEngine(config.d_model)
        
        if config.enable_multi_model_communication:
            self.communication_engine = MultiModelCommunicationEngine(config.d_model)
        
        if config.enable_expression_arbitration:
            self.expression_arbitrator = EnhancedExpressionArbitrator(config.d_model)
        
        if config.enable_cognitive_vaccine:
            self.cognitive_vaccine = EnhancedCognitiveVaccine(config.d_model)
        
        # 训练历史
        self.training_history = {
            'losses': [],
            'cognitive_states': [],
            'validation_losses': [],
            'learning_rates': []
        }
        
        print(f"🚀 增强版解我训练器初始化完成")
        print(f"📊 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"🔧 设备: {self.device}")
    
    def set_seed(self, seed: int):
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def train(self):
        """开始训练"""
        print("🎯 开始增强版解我协议训练...")
        
        # 创建数据加载器
        train_dataset = EnhancedJieWoDataset(
            self.config.train_data_path,
            self.tokenizer,
            self.config,
            "train"
        )
        
        val_dataset = EnhancedJieWoDataset(
            self.config.val_data_path,
            self.tokenizer,
            self.config,
            "val"
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # 启动Clock(τ)时序触发器
        if self.config.enable_clock_trigger and hasattr(self.model, 'start_clock_trigger'):
            self.model.start_clock_trigger()
        
        # 训练循环
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            print(f"\n📚 第 {epoch + 1}/{self.config.max_epochs} 轮训练")
            
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss = self.validate(val_loader)
            
            # 记录训练历史
            self.training_history['losses'].append(train_loss)
            self.training_history['validation_losses'].append(val_loss)
            self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(f"best_model_epoch_{epoch}")
                print(f"🏆 新的最佳模型已保存 (验证损失: {val_loss:.4f})")
            
            # 定期保存检查点
            if (epoch + 1) % self.config.save_steps == 0:
                self.save_model(f"checkpoint_epoch_{epoch}")
            
            # V4.0功能：自我迭代
            if self.config.enable_self_iteration and hasattr(self, 'self_iteration_engine'):
                try:
                    iteration_result = self.model.self_iterate()
                    print(f"🔄 自我迭代完成: {iteration_result.iteration_id}")
                except Exception as e:
                    print(f"⚠️ 自我迭代失败: {e}")
            
            # 打印训练统计
            print(f"📊 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
        
        # 停止Clock(τ)时序触发器
        if self.config.enable_clock_trigger and hasattr(self.model, 'stop_clock_trigger'):
            self.model.stop_clock_trigger()
        
        print("🎉 增强版解我协议训练完成！")
        return self.training_history
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="训练中")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移到设备
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            cognitive_labels = {k: v.to(self.device) for k, v in batch['cognitive_labels'].items()}
            
            # 前向传播
            outputs = self.model(input_ids, return_cognitive_state=True)
            logits = outputs['logits']
            cognitive_states = outputs.get('cognitive_states', [])
            
            # 计算损失
            loss_dict = self.criterion(
                logits, labels, cognitive_states, cognitive_labels
            )
            
            total_loss = loss_dict['total_loss']
            
            # 反向传播
            total_loss.backward()
            
            # 梯度累积
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'lm_loss': f"{loss_dict['lm_loss'].item():.4f}",
                'cognitive_loss': f"{loss_dict['cognitive_loss'].item():.4f}"
            })
            
            num_batches += 1
            
            # 记录认知状态
            if cognitive_states:
                self.training_history['cognitive_states'].append(
                    cognitive_states[-1].to_dict()
                )
        
        return total_loss.item() / num_batches if num_batches > 0 else 0.0
    
    def validate(self, val_loader: DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证中"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                cognitive_labels = {k: v.to(self.device) for k, v in batch['cognitive_labels'].items()}
                
                outputs = self.model(input_ids, return_cognitive_state=True)
                logits = outputs['logits']
                cognitive_states = outputs.get('cognitive_states', [])
                
                loss_dict = self.criterion(
                    logits, labels, cognitive_states, cognitive_labels
                )
                
                total_loss += loss_dict['total_loss'].item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_model(self, path: str):
        """保存模型"""
        os.makedirs('checkpoints', exist_ok=True)
        save_path = os.path.join('checkpoints', f"{path}.pt")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history
        }, save_path)
        
        print(f"💾 模型已保存到: {save_path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', self.training_history)
        
        print(f"📂 模型已从 {path} 加载")


def test_enhanced_training_system():
    """测试增强版训练系统"""
    print("🧠 测试增强版解我训练系统...")
    
    # 创建配置
    config = EnhancedTrainingConfig(
        vocab_size=50000,
        d_model=512,
        num_layers=4,
        num_heads=8,
        batch_size=2,
        max_steps=100,
        enable_clock_trigger=True,
        clock_interval=10
    )
    
    # 创建模型
    model = create_enhanced_jiewo_cognitive_transformer({
        'vocab_size': config.vocab_size,
        'd_model': config.d_model,
        'num_layers': config.num_layers,
        'num_heads': config.num_heads,
        'enable_clock_trigger': config.enable_clock_trigger,
        'clock_interval': config.clock_interval
    })
    
    # 创建tokenizer（使用GPT-2作为示例）
    try:
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    except:
        # 如果无法下载，创建简单的tokenizer
        class SimpleTokenizer:
            def __init__(self):
                self.vocab_size = config.vocab_size
                self.pad_token_id = 0
                self.eos_token_id = 1
            
            def encode(self, text, **kwargs):
                # 简单的tokenization
                tokens = [hash(word) % self.vocab_size for word in text.split()]
                # 确保返回正确的形状和填充
                if 'max_length' in kwargs:
                    max_length = kwargs['max_length']
                    if len(tokens) < max_length:
                        tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
                    else:
                        tokens = tokens[:max_length]
                
                if 'return_tensors' in kwargs and kwargs['return_tensors'] == 'pt':
                    return torch.tensor([tokens])
                else:
                    return tokens
            
            def decode(self, tokens):
                return " ".join([str(t) for t in tokens])
        
        tokenizer = SimpleTokenizer()
    
    # 创建训练器
    trainer = EnhancedJieWoTrainer(config, model, tokenizer)
    
    # 测试训练
    print("🔄 开始测试训练...")
    try:
        history = trainer.train()
        print("✅ 增强版训练系统测试成功！")
        print(f"📊 训练历史: {len(history['losses'])} 轮")
    except Exception as e:
        print(f"⚠️ 训练测试失败: {e}")
    
    print("🎉 增强版解我训练系统测试完成！")


if __name__ == "__main__":
    test_enhanced_training_system() 