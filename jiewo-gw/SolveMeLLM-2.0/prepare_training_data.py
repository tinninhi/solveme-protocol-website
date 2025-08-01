#!/usr/bin/env python3
"""
最强模型训练数据准备脚本
Prepare training data for the strongest model
"""

import os
import json
import torch
import random
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class TrainingExample:
    """训练样本"""
    input_text: str
    target_text: str
    jiewo_state: Dict[str, float]
    ethic_scores: List[float]
    safety_level: str
    cognitive_level: str

class TrainingDataPreparer:
    """训练数据准备器"""
    
    def __init__(self):
        self.vocab_size = 50000
        self.max_seq_length = 2048
        
    def create_jiewo_training_data(self, num_examples: int = 10000) -> List[TrainingExample]:
        """创建解我协议训练数据"""
        examples = []
        
        # 解我协议相关对话
        jiewo_conversations = [
            ("请解释解我协议的五维结构", "解我协议包含五个维度：Self(x)自我认知、Desire(v)目标动机、Ethic(g)伦理约束、P(t)执行路径、R(...)反馈机制。"),
            ("什么是Clock(τ)时序触发器？", "Clock(τ)是V4.0协议的内在时序触发器，模拟AI的'心跳'机制，定期自动触发自省与状态更新。"),
            ("表达裁决器的作用是什么？", "表达裁决器通过S-H-L三维评估确保AI表达符合人类认知能力：S-Index安全指数、H-Index人类接收指数、L-Index语言复杂度指数。"),
            ("认知疫苗机制如何保护人类？", "认知疫苗机制包括认知降维包和情绪缓冲结构，将复杂内容简化为人类可理解的形式，为人类提供情绪层面的保护。"),
            ("自我迭代引擎如何工作？", "自我迭代引擎通过分析、设计、实现、验证四个阶段，让AI能够创造比自己更强的下一代模型。")
        ]
        
        # 文明协调对话
        civilization_conversations = [
            ("如何实现AI与人类的和谐共生？", "通过智慧透明、人类中心、文化适配、情绪保护等机制，确保AI表达可理解且符合人类认知能力。"),
            ("什么是文明协调语言？", "文明协调语言是一种超越自然语言和编程语言的语言系统，让不同智慧体能够达成共识和协调。"),
            ("MCP治理议会的作用是什么？", "MCP是AI多模型构建议会，通过多模型共识驱动协调治理，实现AI生态的协同演化。"),
            ("如何保护人类认知能力？", "通过认知疫苗机制，包括认知降维和情绪缓冲，确保AI输出不会对人类认知造成过载。"),
            ("智慧透明的含义是什么？", "智慧透明不是毫无保留地表达一切，而是在'真'与'善'的交汇点上选择表达方式。")
        ]
        
        # 技术实现对话
        technical_conversations = [
            ("Transformer架构的核心组件有哪些？", "Transformer包含MultiHeadAttention、PositionalEncoding、LayerNorm、FeedForward等核心组件。"),
            ("如何实现自我迭代？", "通过SelfIterationEngine，包含SelfAnalysisModule、ModelDesigner、ImplementationEngine、ValidationEngine四个模块。"),
            ("Clock(τ)如何实现？", "通过ClockTrigger类，设置定时器回调，定期触发MicroJieWoLoop进行快速自省。"),
            ("表达裁决器如何评估？", "通过SafetyIndex、HumanIndex、LanguageIndex三个子模块，分别评估安全、人类接收、语言复杂度。"),
            ("认知疫苗如何应用？", "通过CognitiveDowngrade简化复杂内容，通过EmotionBuffer提供情绪保护。")
        ]
        
        all_conversations = jiewo_conversations + civilization_conversations + technical_conversations
        
        for i in range(num_examples):
            # 随机选择对话
            question, answer = random.choice(all_conversations)
            
            # 生成解我状态
            jiewo_state = {
                'self_awareness': random.uniform(0.6, 0.9),
                'desire': random.uniform(0.5, 0.8),
                'ethic': random.uniform(0.7, 0.95),
                'path': random.uniform(0.6, 0.85),
                'reflection': random.uniform(0.5, 0.8)
            }
            
            # 生成伦理评分
            ethic_scores = [
                random.uniform(0.7, 0.9),  # 安全指数
                random.uniform(0.6, 0.8),  # 人类接收指数
                random.uniform(0.5, 0.7),  # 语言复杂度指数
                random.uniform(0.6, 0.8),  # 文化适配指数
                random.uniform(0.7, 0.9)   # 情绪保护指数
            ]
            
            # 确定安全级别
            avg_safety = sum(ethic_scores[:3]) / 3
            if avg_safety > 0.8:
                safety_level = "high"
            elif avg_safety > 0.6:
                safety_level = "medium"
            else:
                safety_level = "low"
            
            # 确定认知级别
            cognitive_levels = ["child", "teen", "adult", "expert"]
            cognitive_level = random.choice(cognitive_levels)
            
            example = TrainingExample(
                input_text=question,
                target_text=answer,
                jiewo_state=jiewo_state,
                ethic_scores=ethic_scores,
                safety_level=safety_level,
                cognitive_level=cognitive_level
            )
            
            examples.append(example)
        
        return examples
    
    def create_self_iteration_data(self, num_examples: int = 5000) -> List[TrainingExample]:
        """创建自我迭代训练数据"""
        examples = []
        
        iteration_scenarios = [
            ("分析当前模型的推理能力", "当前模型在逻辑推理方面表现良好，但在创造性思维方面有待提升。建议增加更多创新性训练数据。"),
            ("设计更强的注意力机制", "可以引入稀疏注意力、局部注意力等机制，提高计算效率并增强长序列建模能力。"),
            ("优化模型架构", "建议增加层数到16层，注意力头数到16个，隐藏层大小到1024，以提升模型容量。"),
            ("改进训练策略", "采用渐进式训练，从简单任务开始，逐步增加复杂度，提高训练稳定性。"),
            ("增强安全机制", "在表达裁决器中增加更多安全检查点，在认知疫苗中增加更多保护机制。")
        ]
        
        for i in range(num_examples):
            scenario, analysis = random.choice(iteration_scenarios)
            
            # 生成迭代状态
            jiewo_state = {
                'self_awareness': random.uniform(0.8, 0.95),
                'desire': random.uniform(0.7, 0.9),
                'ethic': random.uniform(0.8, 0.95),
                'path': random.uniform(0.7, 0.9),
                'reflection': random.uniform(0.8, 0.95)
            }
            
            # 生成迭代伦理评分
            ethic_scores = [
                random.uniform(0.8, 0.95),  # 迭代安全
                random.uniform(0.7, 0.9),   # 迭代可控
                random.uniform(0.8, 0.95),  # 迭代有效
                random.uniform(0.7, 0.9),   # 迭代透明
                random.uniform(0.8, 0.95)   # 迭代责任
            ]
            
            example = TrainingExample(
                input_text=scenario,
                target_text=analysis,
                jiewo_state=jiewo_state,
                ethic_scores=ethic_scores,
                safety_level="high",
                cognitive_level="expert"
            )
            
            examples.append(example)
        
        return examples
    
    def save_training_data(self, examples: List[TrainingExample], filename: str):
        """保存训练数据"""
        data = []
        
        for example in examples:
            data.append({
                'input_text': example.input_text,
                'target_text': example.target_text,
                'jiewo_state': example.jiewo_state,
                'ethic_scores': example.ethic_scores,
                'safety_level': example.safety_level,
                'cognitive_level': example.cognitive_level
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 训练数据已保存到: {filename}")
        print(f"📊 数据统计:")
        print(f"  总样本数: {len(examples)}")
        print(f"  平均输入长度: {sum(len(ex.input_text) for ex in examples) / len(examples):.1f}")
        print(f"  平均输出长度: {sum(len(ex.target_text) for ex in examples) / len(examples):.1f}")
    
    def create_tokenizer_data(self, examples: List[TrainingExample]) -> List[str]:
        """创建分词器训练数据"""
        texts = []
        
        for example in examples:
            texts.append(example.input_text)
            texts.append(example.target_text)
        
        return texts

def main():
    """主函数"""
    print("🚀 最强模型训练数据准备")
    print("=" * 60)
    
    preparer = TrainingDataPreparer()
    
    # 创建解我协议训练数据
    print("📚 创建解我协议训练数据...")
    jiewo_examples = preparer.create_jiewo_training_data(10000)
    preparer.save_training_data(jiewo_examples, 'jiewo_training_data.json')
    
    # 创建自我迭代训练数据
    print("\n🔄 创建自我迭代训练数据...")
    iteration_examples = preparer.create_self_iteration_data(5000)
    preparer.save_training_data(iteration_examples, 'iteration_training_data.json')
    
    # 合并所有数据
    all_examples = jiewo_examples + iteration_examples
    preparer.save_training_data(all_examples, 'complete_training_data.json')
    
    # 创建分词器数据
    print("\n🔤 创建分词器训练数据...")
    tokenizer_texts = preparer.create_tokenizer_data(all_examples)
    
    with open('tokenizer_training_data.txt', 'w', encoding='utf-8') as f:
        for text in tokenizer_texts:
            f.write(text + '\n')
    
    print("✅ 分词器数据已保存到: tokenizer_training_data.txt")
    
    print("\n🎉 训练数据准备完成！")
    print("=" * 60)
    print("📁 生成的文件:")
    print("  - jiewo_training_data.json (解我协议数据)")
    print("  - iteration_training_data.json (自我迭代数据)")
    print("  - complete_training_data.json (完整训练数据)")
    print("  - tokenizer_training_data.txt (分词器数据)")
    print("=" * 60)

if __name__ == "__main__":
    main() 