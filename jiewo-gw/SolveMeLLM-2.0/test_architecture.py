#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试解我认知架构
"""

import torch
import torch.nn as nn

print("🧠 开始测试解我认知架构...")

# 创建一个简化的解我认知Transformer
class SimpleJieWoCognitiveTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.get('vocab_size', 50000)
        self.d_model = config.get('d_model', 768)
        self.num_heads = config.get('num_heads', 12)
        
        # 简化的嵌入层
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # 简化的注意力层
        self.attention = nn.MultiheadAttention(self.d_model, self.num_heads, batch_first=True)
        
        # 输出层
        self.output = nn.Linear(self.d_model, self.vocab_size)
        
        # 认知状态缓存
        self.cognitive_state_cache = None
    
    def forward(self, input_ids: torch.Tensor, return_cognitive_state: bool = False):
        # 简化的前向传播
        embeddings = self.embedding(input_ids)
        output, attention_weights = self.attention(embeddings, embeddings, embeddings)
        logits = self.output(output)
        
        # 创建简化的认知状态
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        if self.cognitive_state_cache is None:
            self.cognitive_state_cache = {
                'self_awareness': torch.randn(batch_size, self.d_model, device=device),
                'desire_vector': torch.randn(batch_size, self.d_model, device=device),
                'ethic_constraints': torch.randn(batch_size, self.d_model, device=device),
                'execution_path': torch.randn(batch_size, self.d_model, device=device),
                'reflection_feedback': torch.randn(batch_size, self.d_model, device=device),
                'cognitive_confidence': 0.8,
                'evolution_step': 0
            }
        
        output_dict = {
            'logits': logits,
            'hidden_states': output
        }
        
        if return_cognitive_state:
            output_dict['cognitive_state'] = self.cognitive_state_cache
        
        return output_dict
    
    def get_cognitive_state(self):
        return self.cognitive_state_cache
    
    def analyze_cognitive_state(self):
        if self.cognitive_state_cache is None:
            return {"error": "No cognitive state available"}
        
        state = self.cognitive_state_cache
        
        # 计算各维度的强度
        self_strength = torch.norm(state['self_awareness'], dim=-1).mean().item()
        desire_strength = torch.norm(state['desire_vector'], dim=-1).mean().item()
        ethic_strength = torch.norm(state['ethic_constraints'], dim=-1).mean().item()
        path_strength = torch.norm(state['execution_path'], dim=-1).mean().item()
        reflection_strength = torch.norm(state['reflection_feedback'], dim=-1).mean().item()
        
        return {
            "self_awareness_strength": self_strength,
            "desire_strength": desire_strength,
            "ethic_strength": ethic_strength,
            "path_strength": path_strength,
            "reflection_strength": reflection_strength,
            "cognitive_confidence": state['cognitive_confidence'],
            "evolution_step": state['evolution_step'],
            "overall_confidence": (self_strength + desire_strength + ethic_strength + path_strength + reflection_strength) / 5
        }

def count_parameters(model: nn.Module):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }

def test_jiewo_cognitive_architecture():
    """测试解我认知架构"""
    print("🧠 测试解我认知架构...")
    
    # 模型配置
    config = {
        'vocab_size': 50000,
        'd_model': 512,
        'num_layers': 4,
        'num_heads': 8,
        'd_ff': 2048,
        'max_seq_length': 1024,
        'dropout': 0.1
    }
    
    # 创建模型
    model = SimpleJieWoCognitiveTransformer(config)
    
    # 计算参数数量
    param_counts = count_parameters(model)
    print(f"📊 模型参数统计:")
    print(f"  总参数: {param_counts['total_parameters']:,}")
    print(f"  可训练参数: {param_counts['trainable_parameters']:,}")
    print(f"  不可训练参数: {param_counts['non_trainable_parameters']:,}")
    
    # 测试设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 模拟输入
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len)).to(device)
    
    # 前向传播测试
    print("🔄 测试前向传播...")
    outputs = model(input_ids, return_cognitive_state=True)
    
    print(f"📈 输出形状:")
    print(f"  logits: {outputs['logits'].shape}")
    print(f"  hidden_states: {outputs['hidden_states'].shape}")
    
    # 认知状态分析
    print("🧠 测试认知状态分析...")
    analysis = model.analyze_cognitive_state()
    
    print(f"🧠 认知状态分析:")
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("🎉 解我认知架构测试完成！")

if __name__ == "__main__":
    test_jiewo_cognitive_architecture() 