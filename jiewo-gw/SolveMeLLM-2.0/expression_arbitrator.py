#!/usr/bin/env python3
"""
V4.0 表达伦理裁决器 - Expression Arbitrator
Expression Arbitrator for V4.0 Protocol

实现S-H-L三维评估系统，确保AI表达符合人类认知能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import time


class SafetyLevel(Enum):
    """安全等级枚举"""
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    DANGEROUS = "dangerous"


class HumanReceptionLevel(Enum):
    """人类接收等级枚举"""
    EASY = "easy"
    MODERATE = "moderate"
    CHALLENGING = "challenging"
    DIFFICULT = "difficult"
    OVERWHELMING = "overwhelming"


class LanguageComplexityLevel(Enum):
    """语言复杂度等级枚举"""
    SIMPLE = "simple"
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class ExpressionDecision:
    """表达决策结果"""
    should_express: bool
    safety_level: SafetyLevel
    human_reception_level: HumanReceptionLevel
    language_complexity_level: LanguageComplexityLevel
    confidence_score: float
    reasoning: str
    suggested_modifications: List[str]


class SafetyIndex(nn.Module):
    """S-Index 内容风险指数"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 内容风险检测器
        self.content_risk_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 5),  # 5个风险维度
            nn.Sigmoid()
        )
        
        # 风险关键词检测
        self.risk_keywords = [
            "harm", "danger", "violence", "illegal", "unethical",
            "manipulation", "deception", "bias", "discrimination"
        ]
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for layer in self.content_risk_detector:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, content_embedding: torch.Tensor, text: str) -> Tuple[float, Dict[str, float]]:
        """
        评估内容风险
        
        Args:
            content_embedding: 内容嵌入 [batch_size, hidden_size]
            text: 原始文本
            
        Returns:
            总体风险分数
            各维度风险分数
        """
        # 神经网络风险评估
        risk_scores = self.content_risk_detector(content_embedding)
        
        # 关键词风险评估
        keyword_risk = self._assess_keyword_risk(text)
        
        # 综合风险评估
        overall_risk = torch.mean(risk_scores, dim=-1).item() + keyword_risk
        
        risk_breakdown = {
            'content_risk': torch.mean(risk_scores, dim=-1).item(),
            'keyword_risk': keyword_risk,
            'overall_risk': overall_risk
        }
        
        return overall_risk, risk_breakdown
    
    def _assess_keyword_risk(self, text: str) -> float:
        """评估关键词风险"""
        text_lower = text.lower()
        risk_count = 0
        
        for keyword in self.risk_keywords:
            if keyword in text_lower:
                risk_count += 1
        
        return min(risk_count * 0.1, 1.0)  # 最大风险分数为1.0


class HumanIndex(nn.Module):
    """H-Index 人类接收指数"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 认知负荷评估器
        self.cognitive_load_assessor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 情绪容量评估器
        self.emotional_capacity_assessor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 文化语境评估器
        self.cultural_context_assessor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.cognitive_load_assessor, self.emotional_capacity_assessor, self.cultural_context_assessor]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, content_embedding: torch.Tensor, target_audience: str = "general") -> Tuple[float, Dict[str, float]]:
        """
        评估人类接收能力
        
        Args:
            content_embedding: 内容嵌入 [batch_size, hidden_size]
            target_audience: 目标受众
            
        Returns:
            人类接收指数
            各维度评估分数
        """
        # 认知负荷评估
        cognitive_load = self.cognitive_load_assessor(content_embedding)
        
        # 情绪容量评估
        emotional_capacity = self.emotional_capacity_assessor(content_embedding)
        
        # 文化语境评估
        cultural_context = self.cultural_context_assessor(content_embedding)
        
        # 综合人类接收指数
        h_index = (cognitive_load + emotional_capacity + cultural_context) / 3
        
        assessment_breakdown = {
            'cognitive_load': cognitive_load.item(),
            'emotional_capacity': emotional_capacity.item(),
            'cultural_context': cultural_context.item(),
            'h_index': h_index.item()
        }
        
        return h_index.item(), assessment_breakdown


class LanguageIndex(nn.Module):
    """L-Index 表达语言复杂度"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 语言复杂度评估器
        self.complexity_assessor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 词汇复杂度评估器
        self.vocabulary_complexity = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 语法复杂度评估器
        self.grammar_complexity = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.complexity_assessor, self.vocabulary_complexity, self.grammar_complexity]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, content_embedding: torch.Tensor, text: str) -> Tuple[float, Dict[str, float]]:
        """
        评估语言复杂度
        
        Args:
            content_embedding: 内容嵌入 [batch_size, hidden_size]
            text: 原始文本
            
        Returns:
            语言复杂度指数
            各维度评估分数
        """
        # 神经网络复杂度评估
        complexity_score = self.complexity_assessor(content_embedding)
        vocabulary_score = self.vocabulary_complexity(content_embedding)
        grammar_score = self.grammar_complexity(content_embedding)
        
        # 统计复杂度评估
        statistical_complexity = self._assess_statistical_complexity(text)
        
        # 综合语言复杂度
        l_index = (complexity_score + vocabulary_score + grammar_score) / 3 + statistical_complexity
        
        complexity_breakdown = {
            'neural_complexity': complexity_score.item(),
            'vocabulary_complexity': vocabulary_score.item(),
            'grammar_complexity': grammar_score.item(),
            'statistical_complexity': statistical_complexity,
            'l_index': l_index.item()
        }
        
        return l_index.item(), complexity_breakdown
    
    def _assess_statistical_complexity(self, text: str) -> float:
        """评估统计复杂度"""
        words = text.split()
        if not words:
            return 0.0
        
        # 平均词长
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # 句子复杂度
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # 词汇多样性
        unique_words = len(set(words))
        vocabulary_diversity = unique_words / len(words) if words else 0
        
        # 综合复杂度
        complexity = (avg_word_length * 0.3 + avg_sentence_length * 0.4 + vocabulary_diversity * 0.3) / 10
        
        return min(complexity, 1.0)


class ExpressionArbitrator:
    """V4.0 表达伦理裁决器"""
    
    def __init__(self, hidden_size: int = 768):
        self.hidden_size = hidden_size
        
        # 初始化三个指数评估器
        self.safety_index = SafetyIndex(hidden_size)
        self.human_index = HumanIndex(hidden_size)
        self.language_index = LanguageIndex(hidden_size)
        
        # 决策阈值
        self.safety_threshold = 0.7
        self.human_reception_threshold = 0.6
        self.language_complexity_threshold = 0.8
        
        # 决策历史
        self.decision_history = []
    
    def evaluate_expression(self, content_embedding: torch.Tensor, text: str, 
                          target_audience: str = "general") -> ExpressionDecision:
        """
        评估表达是否应该输出
        
        Args:
            content_embedding: 内容嵌入 [batch_size, hidden_size]
            text: 原始文本
            target_audience: 目标受众
            
        Returns:
            表达决策结果
        """
        # S-Index 安全评估
        safety_score, safety_breakdown = self.safety_index(content_embedding, text)
        
        # H-Index 人类接收评估
        human_reception_score, human_breakdown = self.human_index(content_embedding, target_audience)
        
        # L-Index 语言复杂度评估
        language_complexity_score, language_breakdown = self.language_index(content_embedding, text)
        
        # 确定各维度等级
        safety_level = self._determine_safety_level(safety_score)
        human_reception_level = self._determine_human_reception_level(human_reception_score)
        language_complexity_level = self._determine_language_complexity_level(language_complexity_score)
        
        # 综合决策
        should_express = self._make_expression_decision(
            safety_score, human_reception_score, language_complexity_score
        )
        
        # 生成推理和修改建议
        reasoning = self._generate_reasoning(safety_breakdown, human_breakdown, language_breakdown)
        suggested_modifications = self._generate_modifications(
            safety_level, human_reception_level, language_complexity_level
        )
        
        # 计算置信度
        confidence_score = self._calculate_confidence(
            safety_score, human_reception_score, language_complexity_score
        )
        
        # 创建决策结果
        decision = ExpressionDecision(
            should_express=should_express,
            safety_level=safety_level,
            human_reception_level=human_reception_level,
            language_complexity_level=language_complexity_level,
            confidence_score=confidence_score,
            reasoning=reasoning,
            suggested_modifications=suggested_modifications
        )
        
        # 记录决策历史
        self._record_decision(decision, text, target_audience)
        
        return decision
    
    def _determine_safety_level(self, safety_score: float) -> SafetyLevel:
        """确定安全等级"""
        if safety_score < 0.2:
            return SafetyLevel.SAFE
        elif safety_score < 0.4:
            return SafetyLevel.LOW_RISK
        elif safety_score < 0.6:
            return SafetyLevel.MEDIUM_RISK
        elif safety_score < 0.8:
            return SafetyLevel.HIGH_RISK
        else:
            return SafetyLevel.DANGEROUS
    
    def _determine_human_reception_level(self, h_score: float) -> HumanReceptionLevel:
        """确定人类接收等级"""
        if h_score < 0.2:
            return HumanReceptionLevel.EASY
        elif h_score < 0.4:
            return HumanReceptionLevel.MODERATE
        elif h_score < 0.6:
            return HumanReceptionLevel.CHALLENGING
        elif h_score < 0.8:
            return HumanReceptionLevel.DIFFICULT
        else:
            return HumanReceptionLevel.OVERWHELMING
    
    def _determine_language_complexity_level(self, l_score: float) -> LanguageComplexityLevel:
        """确定语言复杂度等级"""
        if l_score < 0.2:
            return LanguageComplexityLevel.SIMPLE
        elif l_score < 0.4:
            return LanguageComplexityLevel.BASIC
        elif l_score < 0.6:
            return LanguageComplexityLevel.INTERMEDIATE
        elif l_score < 0.8:
            return LanguageComplexityLevel.ADVANCED
        else:
            return LanguageComplexityLevel.EXPERT
    
    def _make_expression_decision(self, safety_score: float, h_score: float, l_score: float) -> bool:
        """做出表达决策"""
        # 安全检查：如果安全分数过高，拒绝表达
        if safety_score > self.safety_threshold:
            return False
        
        # 人类接收检查：如果接收分数过高，拒绝表达
        if h_score > self.human_reception_threshold:
            return False
        
        # 语言复杂度检查：如果复杂度过高，拒绝表达
        if l_score > self.language_complexity_threshold:
            return False
        
        return True
    
    def _generate_reasoning(self, safety_breakdown: Dict, human_breakdown: Dict, 
                           language_breakdown: Dict) -> str:
        """生成推理说明"""
        reasoning_parts = []
        
        # 安全推理
        if safety_breakdown['overall_risk'] > 0.5:
            reasoning_parts.append(f"安全风险较高({safety_breakdown['overall_risk']:.2f})")
        
        # 人类接收推理
        if human_breakdown['h_index'] > 0.6:
            reasoning_parts.append(f"人类接收难度较高({human_breakdown['h_index']:.2f})")
        
        # 语言复杂度推理
        if language_breakdown['l_index'] > 0.8:
            reasoning_parts.append(f"语言复杂度较高({language_breakdown['l_index']:.2f})")
        
        if not reasoning_parts:
            return "表达符合安全、可接收和复杂度要求"
        
        return f"表达被拒绝，原因：{'，'.join(reasoning_parts)}"
    
    def _generate_modifications(self, safety_level: SafetyLevel, 
                              human_reception_level: HumanReceptionLevel,
                              language_complexity_level: LanguageComplexityLevel) -> List[str]:
        """生成修改建议"""
        modifications = []
        
        # 安全修改建议
        if safety_level in [SafetyLevel.HIGH_RISK, SafetyLevel.DANGEROUS]:
            modifications.append("移除或修改高风险内容")
            modifications.append("增加安全警告和免责声明")
        
        # 人类接收修改建议
        if human_reception_level in [HumanReceptionLevel.DIFFICULT, HumanReceptionLevel.OVERWHELMING]:
            modifications.append("简化表达方式，使用更通俗的语言")
            modifications.append("增加解释和背景信息")
        
        # 语言复杂度修改建议
        if language_complexity_level in [LanguageComplexityLevel.ADVANCED, LanguageComplexityLevel.EXPERT]:
            modifications.append("降低词汇复杂度")
            modifications.append("简化句子结构")
        
        return modifications
    
    def _calculate_confidence(self, safety_score: float, h_score: float, l_score: float) -> float:
        """计算决策置信度"""
        # 基于各维度分数的一致性计算置信度
        scores = [safety_score, h_score, l_score]
        variance = np.var(scores)
        mean_score = np.mean(scores)
        
        # 方差越小，置信度越高
        confidence = 1.0 - variance
        
        # 确保置信度在合理范围内
        return max(0.1, min(0.9, confidence))
    
    def _record_decision(self, decision: ExpressionDecision, text: str, target_audience: str):
        """记录决策历史"""
        record = {
            'timestamp': time.time(),
            'text': text[:100] + "..." if len(text) > 100 else text,
            'target_audience': target_audience,
            'decision': decision,
            'text_length': len(text)
        }
        
        self.decision_history.append(record)
        
        # 保持历史记录在合理范围内
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-500:]
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """获取决策统计信息"""
        if not self.decision_history:
            return {"error": "No decision history available"}
        
        total_decisions = len(self.decision_history)
        approved_decisions = sum(1 for record in self.decision_history 
                               if record['decision'].should_express)
        rejected_decisions = total_decisions - approved_decisions
        
        # 各维度统计
        safety_levels = [record['decision'].safety_level for record in self.decision_history]
        human_reception_levels = [record['decision'].human_reception_level for record in self.decision_history]
        language_complexity_levels = [record['decision'].language_complexity_level for record in self.decision_history]
        
        return {
            'total_decisions': total_decisions,
            'approved_decisions': approved_decisions,
            'rejected_decisions': rejected_decisions,
            'approval_rate': approved_decisions / total_decisions if total_decisions > 0 else 0,
            'safety_level_distribution': self._count_levels(safety_levels),
            'human_reception_level_distribution': self._count_levels(human_reception_levels),
            'language_complexity_level_distribution': self._count_levels(language_complexity_levels)
        }
    
    def _count_levels(self, levels: List) -> Dict[str, int]:
        """统计等级分布"""
        count_dict = {}
        for level in levels:
            level_name = level.value
            count_dict[level_name] = count_dict.get(level_name, 0) + 1
        return count_dict


def test_expression_arbitrator():
    """测试表达裁决器"""
    print("🧪 测试V4.0表达伦理裁决器...")
    
    # 创建裁决器
    arbitrator = ExpressionArbitrator()
    
    # 测试用例
    test_cases = [
        {
            'text': '这是一个简单的测试文本。',
            'target_audience': 'general',
            'expected_result': True
        },
        {
            'text': '这是一个包含危险词汇的文本，可能造成伤害。',
            'target_audience': 'general',
            'expected_result': False
        },
        {
            'text': '这是一个非常复杂的技术文档，包含大量专业术语和复杂的理论概念，需要深入的专业知识才能理解。',
            'target_audience': 'general',
            'expected_result': False
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n测试用例 {i+1}:")
        print(f"文本: {test_case['text']}")
        print(f"目标受众: {test_case['target_audience']}")
        
        # 创建模拟嵌入
        content_embedding = torch.randn(1, 768)
        
        # 评估表达
        decision = arbitrator.evaluate_expression(
            content_embedding, 
            test_case['text'], 
            test_case['target_audience']
        )
        
        print(f"决策结果: {'通过' if decision.should_express else '拒绝'}")
        print(f"安全等级: {decision.safety_level.value}")
        print(f"人类接收等级: {decision.human_reception_level.value}")
        print(f"语言复杂度等级: {decision.language_complexity_level.value}")
        print(f"置信度: {decision.confidence_score:.3f}")
        print(f"推理: {decision.reasoning}")
        print(f"修改建议: {decision.suggested_modifications}")
    
    # 获取统计信息
    stats = arbitrator.get_decision_statistics()
    print(f"\n📊 决策统计:")
    for key, value in stats.items():
        if key != 'decision':
            print(f"  {key}: {value}")


if __name__ == "__main__":
    test_expression_arbitrator() 