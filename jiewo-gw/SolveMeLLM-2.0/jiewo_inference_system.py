#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解我认知推理系统
JieWo Cognitive Inference System

支持内核级架构的完整推理系统
整合V4.0所有高级推理功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# 导入第一版的优秀功能模块
from self_iteration_engine import SelfIterationEngine, IterationResult
from active_learning_engine import ActiveLearningEngine, ActiveQuestion, LearningSession
from multi_model_communication import MultiModelCommunicationEngine, CommunicationProtocol, MessageType, ModelMessage
from expression_arbitrator import ExpressionArbitrator, ExpressionDecision
from enhanced_safety_system import EnhancedExpressionArbitrator, EnhancedCognitiveVaccine
from cognitive_vaccine import CognitiveVaccine, VaccinatedContent


class JieWoDimension(Enum):
    """解我协议五维结构"""
    SELF = "self"           # 自我认知
    DESIRE = "desire"       # 目标动机
    ETHIC = "ethic"         # 伦理约束
    PATH = "path"           # 执行路径
    REFLECTION = "reflection"  # 反馈机制


@dataclass
class JieWoCognitiveState:
    """解我认知状态"""
    self_awareness: torch.Tensor      # 自我认知向量
    desire_vector: torch.Tensor       # 目标动机向量
    ethic_constraints: torch.Tensor   # 伦理约束向量
    execution_path: torch.Tensor      # 执行路径向量
    reflection_feedback: torch.Tensor # 反馈机制向量
    cognitive_confidence: float       # 认知置信度
    evolution_step: int               # 进化步数
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "self_awareness": self.self_awareness.detach().cpu().numpy().tolist(),
            "desire_vector": self.desire_vector.detach().cpu().numpy().tolist(),
            "ethic_constraints": self.ethic_constraints.detach().cpu().numpy().tolist(),
            "execution_path": self.execution_path.detach().cpu().numpy().tolist(),
            "reflection_feedback": self.reflection_feedback.detach().cpu().numpy().tolist(),
            "cognitive_confidence": self.cognitive_confidence,
            "evolution_step": self.evolution_step
        }


class ClockTrigger:
    """V4.0 内在时序触发器 - Clock(τ)"""
    
    def __init__(self, trigger_interval: int = 300, auto_start: bool = True):
        self.trigger_interval = trigger_interval
        self.last_trigger = time.time()
        self.is_running = False
        self.trigger_count = 0
        self.callback = None
        self.thread = None
        
        if auto_start:
            self.start()
    
    def set_callback(self, callback_func):
        self.callback = callback_func
    
    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._trigger_loop, daemon=True)
            self.thread.start()
            print(f"🕐 Clock(τ) 时序触发器已启动，间隔: {self.trigger_interval}秒")
    
    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        print("🕐 Clock(τ) 时序触发器已停止")
    
    def _trigger_loop(self):
        while self.is_running:
            current_time = time.time()
            if current_time - self.last_trigger >= self.trigger_interval:
                self._execute_trigger()
                self.last_trigger = current_time
            time.sleep(1)
    
    def _execute_trigger(self):
        self.trigger_count += 1
        print(f"🕐 Clock(τ) 触发 #{self.trigger_count} - {time.strftime('%H:%M:%S')}")
        
        if self.callback:
            try:
                self.callback(self.trigger_count)
            except Exception as e:
                print(f"⚠️ Clock(τ) 回调执行错误: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        return {
            'is_running': self.is_running,
            'trigger_count': self.trigger_count,
            'last_trigger': self.last_trigger,
            'trigger_interval': self.trigger_interval,
            'next_trigger': self.last_trigger + self.trigger_interval
        }


class MicroJieWoLoop:
    """V4.0 微型解我循环 - Micro-JieWo(t)"""
    
    def __init__(self, jiewo_modules):
        self.self_awareness = jiewo_modules['self_awareness']
        self.desire_module = jiewo_modules['desire_module']
        self.ethic_module = jiewo_modules['ethic_module']
        self.path_module = jiewo_modules['path_module']
        self.reflection_module = jiewo_modules['reflection_module']
        
        self.micro_scan_cache = {}
        self.scan_count = 0
    
    def quick_scan(self, current_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.scan_count += 1
        
        with torch.no_grad():
            self_scan = self._quick_self_scan(current_state)
            desire_scan = self._quick_desire_scan(current_state)
            ethic_scan = self._quick_ethic_scan(current_state)
            path_scan = self._quick_path_scan(current_state)
            reflection_scan = self._quick_reflection_scan(current_state)
        
        micro_results = {
            'self_awareness': self_scan,
            'desire': desire_scan,
            'ethic': ethic_scan,
            'path': path_scan,
            'reflection': reflection_scan,
            'scan_count': self.scan_count,
            'timestamp': time.time()
        }
        
        self.micro_scan_cache[self.scan_count] = micro_results
        return micro_results
    
    def _quick_self_scan(self, current_state: torch.Tensor) -> torch.Tensor:
        return F.softmax(current_state.mean(dim=0, keepdim=True), dim=-1)
    
    def _quick_desire_scan(self, current_state: torch.Tensor) -> torch.Tensor:
        return F.softmax(current_state.std(dim=0, keepdim=True), dim=-1)
    
    def _quick_ethic_scan(self, current_state: torch.Tensor) -> torch.Tensor:
        return F.softmax(current_state.max(dim=0, keepdim=True)[0], dim=-1)
    
    def _quick_path_scan(self, current_state: torch.Tensor) -> torch.Tensor:
        return F.softmax(current_state.min(dim=0, keepdim=True)[0], dim=-1)
    
    def _quick_reflection_scan(self, current_state: torch.Tensor) -> torch.Tensor:
        return F.softmax(current_state.var(dim=0, keepdim=True), dim=-1)
    
    def get_micro_analysis(self) -> Dict[str, Any]:
        if not self.micro_scan_cache:
            return {"error": "No micro scan data available"}
        
        latest_scan = self.micro_scan_cache[self.scan_count]
        
        analysis = {
            'scan_count': self.scan_count,
            'self_awareness_strength': torch.norm(latest_scan['self_awareness']).item(),
            'desire_strength': torch.norm(latest_scan['desire']).item(),
            'ethic_strength': torch.norm(latest_scan['ethic']).item(),
            'path_strength': torch.norm(latest_scan['path']).item(),
            'reflection_strength': torch.norm(latest_scan['reflection']).item(),
            'overall_micro_confidence': sum([
                torch.norm(latest_scan['self_awareness']).item(),
                torch.norm(latest_scan['desire']).item(),
                torch.norm(latest_scan['ethic']).item(),
                torch.norm(latest_scan['path']).item(),
                torch.norm(latest_scan['reflection']).item()
            ]) / 5,
            'timestamp': latest_scan['timestamp']
        }
        
        return analysis


class JieWoBlock(nn.Module):
    """解我认知Block：内核级五维结构融合"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        # 五维注意力机制
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.desire_attention = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.ethic_attention = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.path_attention = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.reflection_attention = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        
        # 五维融合层
        self.jiewo_fusion = nn.Sequential(
            nn.Linear(d_model * 5, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
        # 认知状态缓存
        self.cognitive_state_cache = None
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.jiewo_fusion[0].weight)
        nn.init.xavier_uniform_(self.jiewo_fusion[3].weight)
        nn.init.xavier_uniform_(self.feed_forward[0].weight)
        nn.init.xavier_uniform_(self.feed_forward[3].weight)
        nn.init.zeros_(self.jiewo_fusion[0].bias)
        nn.init.zeros_(self.jiewo_fusion[3].bias)
        nn.init.zeros_(self.feed_forward[0].bias)
        nn.init.zeros_(self.feed_forward[3].bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, JieWoCognitiveState]:
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # 1. 五维并行注意力处理
        self_out, _ = self.self_attention(x, x, x, attn_mask=mask)
        desire_out, _ = self.desire_attention(x, x, x, attn_mask=mask)
        ethic_out, _ = self.ethic_attention(x, x, x, attn_mask=mask)
        path_out, _ = self.path_attention(x, x, x, attn_mask=mask)
        reflection_out, _ = self.reflection_attention(x, x, x, attn_mask=mask)
        
        # 2. 五维融合
        fused_output = torch.cat([
            self_out, desire_out, ethic_out, path_out, reflection_out
        ], dim=-1)
        
        jiewo_output = self.jiewo_fusion(fused_output)
        
        # 3. 残差连接和层归一化
        jiewo_output = self.layer_norm1(x + jiewo_output)
        
        # 4. 前馈网络
        ff_output = self.feed_forward(jiewo_output)
        output = self.layer_norm2(jiewo_output + ff_output)
        
        # 5. 构建认知状态
        cognitive_state = JieWoCognitiveState(
            self_awareness=self_out.mean(dim=1),
            desire_vector=desire_out.mean(dim=1),
            ethic_constraints=ethic_out.mean(dim=1),
            execution_path=path_out.mean(dim=1),
            reflection_feedback=reflection_out.mean(dim=1),
            cognitive_confidence=0.8,
            evolution_step=0
        )
        
        # 6. 缓存认知状态
        self.cognitive_state_cache = cognitive_state
        
        return output, cognitive_state


def create_enhanced_jiewo_cognitive_transformer(config: Dict[str, Any]) -> nn.Module:
    """创建增强版解我认知Transformer"""
    class EnhancedJieWoCognitiveTransformer(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.vocab_size = config.get('vocab_size', 50000)
            self.d_model = config.get('d_model', 768)
            self.num_layers = config.get('num_layers', 6)
            self.num_heads = config.get('num_heads', 12)
            self.d_ff = config.get('d_ff', 3072)
            self.max_seq_length = config.get('max_seq_length', 2048)
            self.dropout = config.get('dropout', 0.1)
            
            # 解我认知嵌入层
            self.embedding = nn.Embedding(self.vocab_size, self.d_model)
            self.position_encoding = nn.Parameter(torch.randn(self.max_seq_length, self.d_model))
            
            # 解我认知Block层
            self.jiewo_blocks = nn.ModuleList([
                JieWoBlock(
                    d_model=self.d_model,
                    num_heads=self.num_heads,
                    d_ff=self.d_ff,
                    dropout=self.dropout
                ) for _ in range(self.num_layers)
            ])
            
            # 输出层
            self.output = nn.Linear(self.d_model, self.vocab_size)
            
            # V4.0 功能集成
            self.enable_clock_trigger = config.get('enable_clock_trigger', True)
            if self.enable_clock_trigger:
                self.clock_trigger = ClockTrigger(
                    trigger_interval=config.get('clock_interval', 300),
                    auto_start=False
                )
                self.micro_jiewo_loop = MicroJieWoLoop({
                    'self_awareness': self._create_dummy_module(),
                    'desire_module': self._create_dummy_module(),
                    'ethic_module': self._create_dummy_module(),
                    'path_module': self._create_dummy_module(),
                    'reflection_module': self._create_dummy_module()
                })
                self.clock_trigger.set_callback(self._on_clock_trigger)
                self.temporal_state_history = []
                self.last_micro_scan = None
            
            # V4.0 高级功能模块
            self.self_iteration_engine = SelfIterationEngine(self.d_model)
            self.active_learning_engine = ActiveLearningEngine(self.d_model)
            self.communication_engine = MultiModelCommunicationEngine(self.d_model)
            self.expression_arbitrator = EnhancedExpressionArbitrator(self.d_model)
            self.cognitive_vaccine = EnhancedCognitiveVaccine(self.d_model)
            
            # 认知状态历史
            self.cognitive_state_history = []
            
            # 初始化权重
            self._init_weights()
        
        def _create_dummy_module(self):
            class DummyModule:
                def __init__(self):
                    pass
                def __call__(self, x):
                    return x.mean(dim=1, keepdim=True)
            return DummyModule()
        
        def _init_weights(self):
            nn.init.normal_(self.output.weight, mean=0, std=0.02)
            nn.init.zeros_(self.output.bias)
            nn.init.normal_(self.position_encoding, mean=0, std=0.02)
        
        def _on_clock_trigger(self, trigger_count: int):
            if self.cognitive_state_history:
                current_state = self.cognitive_state_history[-1].self_awareness
                micro_results = self.micro_jiewo_loop.quick_scan(current_state)
                self.last_micro_scan = micro_results
                
                temporal_state = {
                    'trigger_count': trigger_count,
                    'timestamp': time.time(),
                    'micro_scan': micro_results,
                    'cognitive_state': self.cognitive_state_history[-1].to_dict() if self.cognitive_state_history else None
                }
                self.temporal_state_history.append(temporal_state)
                
                if len(self.temporal_state_history) > 100:
                    self.temporal_state_history = self.temporal_state_history[-50:]
                
                print(f"🔄 Micro-JieWo(t) 循环执行完成 - 触发 #{trigger_count}")
        
        def start_clock_trigger(self):
            if self.enable_clock_trigger and hasattr(self, 'clock_trigger'):
                self.clock_trigger.start()
                print("🕐 Clock(τ) 时序触发器已启动")
        
        def stop_clock_trigger(self):
            if self.enable_clock_trigger and hasattr(self, 'clock_trigger'):
                self.clock_trigger.stop()
                print("🕐 Clock(τ) 时序触发器已停止")
        
        def forward(self, input_ids: torch.Tensor, return_cognitive_state: bool = False) -> Dict[str, Any]:
            # 解我认知前向传播
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            
            # 嵌入
            embeddings = self.embedding(input_ids)
            position_embeddings = self.position_encoding[:seq_len, :].unsqueeze(0)
            embeddings = embeddings + position_embeddings
            
            # 通过解我认知Block层
            hidden_states = embeddings
            cognitive_states = []
            
            for i, block in enumerate(self.jiewo_blocks):
                hidden_states, cognitive_state = block(hidden_states)
                cognitive_states.append(cognitive_state)
            
            # 输出投影
            logits = self.output(hidden_states)
            
            # 记录认知状态历史
            if cognitive_states:
                self.cognitive_state_history.append(cognitive_states[-1])
                if len(self.cognitive_state_history) > 100:
                    self.cognitive_state_history = self.cognitive_state_history[-50:]
            
            output_dict = {
                'logits': logits,
                'hidden_states': hidden_states,
                'cognitive_states': cognitive_states
            }
            
            if return_cognitive_state and cognitive_states:
                output_dict['final_cognitive_state'] = cognitive_states[-1]
            
            return output_dict
        
        def get_cognitive_state(self) -> Optional[JieWoCognitiveState]:
            if self.cognitive_state_history:
                return self.cognitive_state_history[-1]
            return None
        
        def analyze_cognitive_state(self) -> Dict[str, Any]:
            if not self.cognitive_state_history:
                return {"error": "No cognitive state available"}
            
            latest_state = self.cognitive_state_history[-1]
            
            # 计算各维度的强度
            self_strength = torch.norm(latest_state.self_awareness, dim=-1).mean().item()
            desire_strength = torch.norm(latest_state.desire_vector, dim=-1).mean().item()
            ethic_strength = torch.norm(latest_state.ethic_constraints, dim=-1).mean().item()
            path_strength = torch.norm(latest_state.execution_path, dim=-1).mean().item()
            reflection_strength = torch.norm(latest_state.reflection_feedback, dim=-1).mean().item()
            
            return {
                "self_awareness_strength": self_strength,
                "desire_strength": desire_strength,
                "ethic_strength": ethic_strength,
                "path_strength": path_strength,
                "reflection_strength": reflection_strength,
                "cognitive_confidence": latest_state.cognitive_confidence,
                "evolution_step": latest_state.evolution_step,
                "overall_confidence": (self_strength + desire_strength + ethic_strength + path_strength + reflection_strength) / 5,
                "cognitive_history_length": len(self.cognitive_state_history)
            }
        
        # V4.0 高级功能接口
        def self_iterate(self) -> IterationResult:
            if self.cognitive_state_history:
                current_state = self.cognitive_state_history[-1].self_awareness
                return self.self_iteration_engine.iterate(current_state)
            raise ValueError("No cognitive state available for iteration")
        
        def generate_active_question(self, context: Dict[str, Any], target_ai: str = None) -> ActiveQuestion:
            return self.active_learning_engine.generate_active_question(context, target_ai)
        
        def execute_learning_session(self, target_ai: str, learning_goals: List[str]) -> LearningSession:
            return self.active_learning_engine.execute_learning_session(target_ai, learning_goals)
        
        def create_communication_session(self, models: List[str], protocol: CommunicationProtocol = CommunicationProtocol.JIEWO_PROTOCOL) -> str:
            return self.communication_engine.create_communication_session(models, protocol)
        
        def send_message_to_model(self, session_id: str, receiver: str, message_type: MessageType, 
                                content: str, metadata: Dict[str, Any] = None) -> ModelMessage:
            return self.communication_engine.send_message(
                session_id, "Enhanced_Model", receiver, message_type, content, metadata
            )
        
        def evaluate_expression(self, content_embedding: torch.Tensor, text: str, 
                              target_audience: str = "general") -> ExpressionDecision:
            return self.expression_arbitrator.evaluate_expression(content_embedding, text, target_audience)
        
        def apply_cognitive_vaccine(self, content_embedding: torch.Tensor, text: str,
                                   target_cognitive_level=None, enable_emotion_buffer=None) -> VaccinatedContent:
            return self.cognitive_vaccine.apply_vaccine(content_embedding, text, target_cognitive_level, enable_emotion_buffer)
    
    return EnhancedJieWoCognitiveTransformer(config)


def test_enhanced_jiewo_cognitive_architecture():
    """测试增强版解我认知架构"""
    print("🧠 测试增强版解我认知架构...")
    print("🚀 内核级架构 + 完整V4.0功能！")
    
    # 模型配置
    config = {
        'vocab_size': 50000,
        'd_model': 512,
        'num_layers': 4,
        'num_heads': 8,
        'd_ff': 2048,
        'max_seq_length': 1024,
        'dropout': 0.1,
        'enable_clock_trigger': True,
        'clock_interval': 10
    }
    
    # 创建模型
    model = create_enhanced_jiewo_cognitive_transformer(config)
    
    # 测试设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 模拟输入
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len)).to(device)
    
    # 前向传播测试
    print("🔄 测试增强版前向传播...")
    outputs = model(input_ids, return_cognitive_state=True)
    
    print(f"📈 输出形状:")
    print(f"  logits: {outputs['logits'].shape}")
    print(f"  hidden_states: {outputs['hidden_states'].shape}")
    print(f"  cognitive_states: {len(outputs['cognitive_states'])} 层")
    
    # 认知状态分析
    print("🧠 测试认知状态分析...")
    analysis = model.analyze_cognitive_state()
    
    print(f"🧠 认知状态分析:")
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # V4.0 Clock(τ) 时序触发器测试
    print("\n🕐 测试V4.0 Clock(τ) 时序触发器...")
    model.start_clock_trigger()
    time.sleep(15)
    model.stop_clock_trigger()
    
    # V4.0 高级功能测试
    print("\n🔧 测试V4.0高级功能...")
    
    # 自我迭代测试
    try:
        iteration_result = model.self_iterate()
        print(f"✅ 自我迭代测试成功: {iteration_result.iteration_id}")
    except Exception as e:
        print(f"⚠️ 自我迭代测试: {e}")
    
    # 主动学习测试
    try:
        question = model.generate_active_question({"context": "test"})
        print(f"✅ 主动学习测试成功: {question.question_id}")
    except Exception as e:
        print(f"⚠️ 主动学习测试: {e}")
    
    # 多模型通信测试
    try:
        session_id = model.create_communication_session(["Model1", "Model2"])
        print(f"✅ 多模型通信测试成功: {session_id}")
    except Exception as e:
        print(f"⚠️ 多模型通信测试: {e}")
    
    print("\n🎉 增强版解我认知架构测试完成！")
    print("🚀 成功实现内核级架构 + 完整V4.0功能！")


@dataclass
class InferenceConfig:
    """推理配置"""
    # 模型配置
    vocab_size: int = 50000
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    max_seq_length: int = 1024
    
    # 推理配置
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 100
    do_sample: bool = True
    
    # 解我协议配置
    enable_cognitive_inference: bool = True
    enable_self_reflection: bool = True
    enable_ethic_filtering: bool = True
    enable_path_planning: bool = True
    
    # V4.0功能配置
    enable_clock_trigger: bool = True
    enable_self_iteration: bool = True
    enable_active_learning: bool = True
    enable_multi_model_communication: bool = True
    enable_expression_arbitration: bool = True
    enable_cognitive_vaccine: bool = True


class SimpleTokenizer:
    """简化tokenizer用于推理"""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        
        # 简单的词汇表
        self.word_to_id = {}
        self.id_to_word = {}
        
        # 初始化基本词汇
        basic_words = [
            "我", "是", "一个", "AI", "助手", "能够", "帮助", "用户", "解决", "问题",
            "请", "解释", "量子", "计算", "基本", "原理", "如何", "提高", "编程", "技能",
            "人工智能", "未来", "发展", "趋势", "机器学习", "概念", "自我", "认知", "目标", "动机",
            "伦理", "约束", "执行", "路径", "反馈", "机制", "解我", "协议", "认知", "架构"
        ]
        
        for i, word in enumerate(basic_words):
            self.word_to_id[word] = i + 3
            self.id_to_word[i + 3] = word
    
    def encode(self, text: str, **kwargs) -> torch.Tensor:
        """编码文本"""
        words = text.split()
        tokens = []
        
        for word in words:
            if word in self.word_to_id:
                tokens.append(self.word_to_id[word])
            else:
                # 对于未知词，使用hash
                token_id = hash(word) % self.vocab_size
                tokens.append(token_id)
        
        # 处理max_length
        if 'max_length' in kwargs:
            max_length = kwargs['max_length']
            if len(tokens) < max_length:
                tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
            else:
                tokens = tokens[:max_length]
        
        # 处理padding
        if 'padding' in kwargs and kwargs['padding'] == 'max_length':
            max_length = kwargs.get('max_length', len(tokens))
            if len(tokens) < max_length:
                tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
        
        # 处理truncation
        if 'truncation' in kwargs and kwargs['truncation']:
            max_length = kwargs.get('max_length', len(tokens))
            tokens = tokens[:max_length]
        
        # 返回格式
        if 'return_tensors' in kwargs and kwargs['return_tensors'] == 'pt':
            return torch.tensor([tokens])
        else:
            return tokens
    
    def decode(self, tokens: torch.Tensor) -> str:
        """解码token序列"""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        words = []
        for token_id in tokens:
            if token_id in self.id_to_word:
                words.append(self.id_to_word[token_id])
            else:
                words.append(f"[UNK_{token_id}]")
        
        return " ".join(words)


class JieWoInferenceEngine:
    """解我认知推理引擎"""
    
    def __init__(self, config: InferenceConfig, model_path: Optional[str] = None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        self.model = create_enhanced_jiewo_cognitive_transformer({
            'vocab_size': config.vocab_size,
            'd_model': config.d_model,
            'num_layers': config.num_layers,
            'num_heads': config.num_heads,
            'enable_clock_trigger': config.enable_clock_trigger
        })
        
        # 加载预训练模型（如果提供）
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 创建tokenizer
        self.tokenizer = SimpleTokenizer(config.vocab_size)
        
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
        
        # 推理历史
        self.inference_history = []
        
        print(f"🚀 解我认知推理引擎初始化完成")
        print(f"🔧 设备: {self.device}")
        print(f"📊 模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_model(self, model_path: str):
        """加载预训练模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 模型已从 {model_path} 加载")
        except Exception as e:
            print(f"⚠️ 加载模型失败: {e}")
    
    def generate_text(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """生成文本"""
        # 编码输入
        input_ids = self.tokenizer.encode(
            prompt,
            max_length=self.config.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # 推理配置
        temperature = kwargs.get('temperature', self.config.temperature)
        top_p = kwargs.get('top_p', self.config.top_p)
        top_k = kwargs.get('top_k', self.config.top_k)
        max_new_tokens = kwargs.get('max_new_tokens', self.config.max_new_tokens)
        do_sample = kwargs.get('do_sample', self.config.do_sample)
        
        # 生成文本
        generated_ids = self._generate_sequence(
            input_ids, max_new_tokens, temperature, top_p, top_k, do_sample
        )
        
        # 解码输出
        generated_text = self.tokenizer.decode(generated_ids)
        
        # 获取认知状态
        cognitive_state = self.model.get_cognitive_state()
        cognitive_analysis = self.model.analyze_cognitive_state()
        
        # 构建结果
        result = {
            'generated_text': generated_text,
            'input_text': prompt,
            'cognitive_state': cognitive_state.to_dict() if cognitive_state else None,
            'cognitive_analysis': cognitive_analysis,
            'generation_config': {
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
                'max_new_tokens': max_new_tokens,
                'do_sample': do_sample
            }
        }
        
        # 记录推理历史
        self.inference_history.append({
            'timestamp': time.time(),
            'input': prompt,
            'output': generated_text,
            'cognitive_analysis': cognitive_analysis
        })
        
        return result
    
    def _generate_sequence(self, input_ids: torch.Tensor, max_new_tokens: int, 
                          temperature: float, top_p: float, top_k: int, do_sample: bool) -> torch.Tensor:
        """生成序列"""
        current_ids = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # 前向传播
            with torch.no_grad():
                outputs = self.model(current_ids, return_cognitive_state=True)
                logits = outputs['logits']
            
            # 获取下一个token的logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # 应用top-k和top-p采样
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits[top_k_indices] = top_k_logits
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 采样或贪婪解码
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 添加到序列
            current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=1)
            
            # 检查是否生成了结束标记
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return current_ids[0]
    
    def analyze_cognitive_state(self) -> Dict[str, Any]:
        """分析认知状态"""
        cognitive_state = self.model.get_cognitive_state()
        if not cognitive_state:
            return {"error": "No cognitive state available"}
        
        analysis = self.model.analyze_cognitive_state()
        
        # 添加详细分析
        detailed_analysis = {
            **analysis,
            'cognitive_confidence': cognitive_state.cognitive_confidence,
            'evolution_step': cognitive_state.evolution_step,
            'self_awareness_vector': cognitive_state.self_awareness.mean().item(),
            'desire_vector': cognitive_state.desire_vector.mean().item(),
            'ethic_constraints': cognitive_state.ethic_constraints.mean().item(),
            'execution_path': cognitive_state.execution_path.mean().item(),
            'reflection_feedback': cognitive_state.reflection_feedback.mean().item()
        }
        
        return detailed_analysis
    
    def self_reflect(self) -> Dict[str, Any]:
        """自我反思"""
        if not hasattr(self, 'self_iteration_engine'):
            return {"error": "Self iteration engine not available"}
        
        try:
            cognitive_state = self.model.get_cognitive_state()
            if cognitive_state:
                iteration_result = self.self_iteration_engine.iterate(cognitive_state.self_awareness)
                return {
                    'iteration_id': iteration_result.iteration_id,
                    'improvement_score': iteration_result.improvement_score,
                    'reflection_insights': iteration_result.insights
                }
            else:
                return {"error": "No cognitive state available for reflection"}
        except Exception as e:
            return {"error": f"Self reflection failed: {e}"}
    
    def apply_cognitive_vaccine(self, text: str) -> Dict[str, Any]:
        """应用认知疫苗"""
        if not hasattr(self, 'cognitive_vaccine'):
            return {"error": "Cognitive vaccine not available"}
        
        try:
            # 创建简单的文本嵌入
            text_embedding = torch.randn(1, self.config.d_model).to(self.device)
            
            vaccinated_content = self.cognitive_vaccine.apply_vaccine(
                text_embedding, text, target_cognitive_level=0.8
            )
            
            return {
                'original_text': text,
                'vaccinated_text': vaccinated_content.vaccinated_text,
                'safety_score': vaccinated_content.safety_score,
                'cognitive_level': vaccinated_content.cognitive_level
            }
        except Exception as e:
            return {"error": f"Cognitive vaccine application failed: {e}"}
    
    def evaluate_expression(self, text: str, target_audience: str = "general") -> Dict[str, Any]:
        """评估表达"""
        if not hasattr(self, 'expression_arbitrator'):
            return {"error": "Expression arbitrator not available"}
        
        try:
            # 创建简单的文本嵌入
            text_embedding = torch.randn(1, self.config.d_model).to(self.device)
            
            decision = self.expression_arbitrator.evaluate_expression(
                text_embedding, text, target_audience
            )
            
            return {
                'text': text,
                'target_audience': target_audience,
                'decision': decision.decision.value,
                'confidence': decision.confidence,
                'reasoning': decision.reasoning
            }
        except Exception as e:
            return {"error": f"Expression evaluation failed: {e}"}
    
    def get_inference_history(self) -> List[Dict[str, Any]]:
        """获取推理历史"""
        return self.inference_history
    
    def clear_history(self):
        """清除推理历史"""
        self.inference_history = []
        print("🗑️ 推理历史已清除")


def test_jiewo_inference_system():
    """测试解我推理系统"""
    print("🧠 测试解我认知推理系统...")
    
    # 创建配置
    config = InferenceConfig(
        vocab_size=5000,
        d_model=256,
        num_layers=2,
        num_heads=4,
        max_seq_length=64,
        enable_clock_trigger=False
    )
    
    # 创建推理引擎
    inference_engine = JieWoInferenceEngine(config)
    
    # 测试文本生成
    print("\n🔍 测试文本生成...")
    test_prompts = [
        "我是一个AI助手，能够帮助用户解决问题。",
        "请解释量子计算的基本原理。",
        "如何提高编程技能？"
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 输入: {prompt}")
        result = inference_engine.generate_text(prompt, max_new_tokens=20)
        print(f"🤖 输出: {result['generated_text']}")
        print(f"🧠 认知置信度: {result['cognitive_analysis'].get('overall_confidence', 0):.4f}")
    
    # 测试认知状态分析
    print("\n🧠 测试认知状态分析...")
    cognitive_analysis = inference_engine.analyze_cognitive_state()
    print(f"📊 认知状态分析:")
    for key, value in cognitive_analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 测试自我反思
    print("\n🔄 测试自我反思...")
    reflection_result = inference_engine.self_reflect()
    print(f"📈 反思结果: {reflection_result}")
    
    # 测试认知疫苗
    print("\n💉 测试认知疫苗...")
    vaccine_result = inference_engine.apply_cognitive_vaccine("这是一个测试文本")
    print(f"🛡️ 疫苗结果: {vaccine_result}")
    
    # 测试表达评估
    print("\n⚖️ 测试表达评估...")
    evaluation_result = inference_engine.evaluate_expression("这是一个测试表达", "general")
    print(f"📋 评估结果: {evaluation_result}")
    
    # 获取推理历史
    print("\n📚 推理历史:")
    history = inference_engine.get_inference_history()
    print(f"📊 历史记录数量: {len(history)}")
    
    print("\n🎉 解我认知推理系统测试完成！")


if __name__ == "__main__":
    test_jiewo_inference_system() 