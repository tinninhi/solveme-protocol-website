#!/usr/bin/env python3
"""
V4.0 认知疫苗机制 - Cognitive Vaccine
Cognitive Vaccine Mechanism for V4.0 Protocol

实现认知降维包和情绪缓冲结构，保护人类认知能力
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
import re


class CognitiveLevel(Enum):
    """认知等级枚举"""
    INFANT = "infant"           # 婴儿级
    CHILD = "child"             # 儿童级
    TEEN = "teen"               # 青少年级
    ADULT = "adult"             # 成年级
    EXPERT = "expert"           # 专家级


class EmotionalIntensity(Enum):
    """情绪强度枚举"""
    CALM = "calm"              # 平静
    MILD = "mild"              # 轻微
    MODERATE = "moderate"      # 中等
    INTENSE = "intense"        # 强烈
    OVERWHELMING = "overwhelming"  # 压倒性


@dataclass
class VaccinatedContent:
    """接种疫苗后的内容"""
    original_content: str
    vaccinated_content: str
    cognitive_level: CognitiveLevel
    emotional_intensity: EmotionalIntensity
    vaccine_applied: List[str]
    confidence_score: float


class CognitiveDowngrade(nn.Module):
    """认知降维包"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 认知复杂度评估器
        self.cognitive_complexity_assessor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 认知降维器
        self.cognitive_downgrader = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # 简化词汇映射
        self.simplification_mapping = {
            # 复杂词汇 -> 简单词汇
            "sophisticated": "advanced",
            "elaborate": "detailed", 
            "comprehensive": "complete",
            "theoretical": "concept",
            "empirical": "practical",
            "methodological": "method",
            "analytical": "analysis",
            "systematic": "organized",
            "paradigm": "model",
            "framework": "structure"
        }
        
        # 复杂句式模式
        self.complex_patterns = [
            r"notwithstanding the fact that",
            r"in light of the aforementioned",
            r"it is imperative to note that",
            r"furthermore, it should be emphasized",
            r"consequently, it follows that"
        ]
        
        # 简化句式替换
        self.simple_replacements = [
            "although",
            "given this",
            "it's important that",
            "also,",
            "so,"
        ]
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for layer in self.cognitive_complexity_assessor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        for layer in self.cognitive_downgrader:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, content_embedding: torch.Tensor, text: str, target_level: CognitiveLevel = CognitiveLevel.ADULT) -> Tuple[str, Dict[str, Any]]:
        """
        应用认知降维
        
        Args:
            content_embedding: 内容嵌入 [batch_size, hidden_size]
            text: 原始文本
            target_level: 目标认知等级
            
        Returns:
            降维后的文本
            降维分析信息
        """
        # 评估原始认知复杂度
        original_complexity = self.cognitive_complexity_assessor(content_embedding).item()
        
        # 根据目标等级确定降维强度
        downgrade_intensity = self._get_downgrade_intensity(target_level)
        
        # 应用词汇简化
        simplified_text = self._simplify_vocabulary(text)
        
        # 应用句式简化
        simplified_text = self._simplify_sentences(simplified_text)
        
        # 应用结构简化
        simplified_text = self._simplify_structure(simplified_text)
        
        # 评估降维效果
        downgrade_analysis = {
            'original_complexity': original_complexity,
            'target_level': target_level.value,
            'downgrade_intensity': downgrade_intensity,
            'vocabulary_simplified': len(self._get_simplified_words(text)) > 0,
            'sentences_simplified': len(self._get_simplified_sentences(text)) > 0,
            'structure_simplified': len(self._get_simplified_structures(text)) > 0
        }
        
        return simplified_text, downgrade_analysis
    
    def _get_downgrade_intensity(self, target_level: CognitiveLevel) -> float:
        """获取降维强度"""
        intensity_mapping = {
            CognitiveLevel.INFANT: 0.9,
            CognitiveLevel.CHILD: 0.7,
            CognitiveLevel.TEEN: 0.5,
            CognitiveLevel.ADULT: 0.3,
            CognitiveLevel.EXPERT: 0.1
        }
        return intensity_mapping.get(target_level, 0.3)
    
    def _simplify_vocabulary(self, text: str) -> str:
        """简化词汇"""
        simplified_text = text
        
        # 应用词汇映射
        for complex_word, simple_word in self.simplification_mapping.items():
            pattern = r'\b' + re.escape(complex_word) + r'\b'
            simplified_text = re.sub(pattern, simple_word, simplified_text, flags=re.IGNORECASE)
        
        return simplified_text
    
    def _simplify_sentences(self, text: str) -> str:
        """简化句式"""
        simplified_text = text
        
        # 应用句式简化
        for i, pattern in enumerate(self.complex_patterns):
            if i < len(self.simple_replacements):
                simplified_text = re.sub(pattern, self.simple_replacements[i], simplified_text, flags=re.IGNORECASE)
        
        return simplified_text
    
    def _simplify_structure(self, text: str) -> str:
        """简化结构"""
        # 分割长句
        sentences = text.split('.')
        simplified_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 100:  # 如果句子太长
                # 在逗号处分割
                parts = sentence.split(',')
                if len(parts) > 2:
                    # 取前两部分作为简化句子
                    simplified_sentence = ','.join(parts[:2]) + '.'
                    simplified_sentences.append(simplified_sentence)
                else:
                    simplified_sentences.append(sentence + '.')
            else:
                simplified_sentences.append(sentence + '.')
        
        return ' '.join(simplified_sentences)
    
    def _get_simplified_words(self, text: str) -> List[str]:
        """获取被简化的词汇"""
        simplified_words = []
        for complex_word in self.simplification_mapping.keys():
            if complex_word.lower() in text.lower():
                simplified_words.append(complex_word)
        return simplified_words
    
    def _get_simplified_sentences(self, text: str) -> List[str]:
        """获取被简化的句式"""
        simplified_sentences = []
        for pattern in self.complex_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                simplified_sentences.append(pattern)
        return simplified_sentences
    
    def _get_simplified_structures(self, text: str) -> List[str]:
        """获取被简化的结构"""
        long_sentences = []
        sentences = text.split('.')
        for sentence in sentences:
            if len(sentence.strip()) > 100:
                long_sentences.append(sentence.strip())
        return long_sentences


class EmotionBuffer(nn.Module):
    """情绪缓冲结构"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 情绪强度评估器
        self.emotional_intensity_assessor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 情绪缓冲器
        self.emotion_buffer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # 情绪关键词映射
        self.emotion_keywords = {
            'intense': ['overwhelming', 'devastating', 'terrifying', 'horrific', 'catastrophic'],
            'moderate': ['concerning', 'worrisome', 'troubling', 'disturbing', 'upsetting'],
            'mild': ['sad', 'disappointing', 'frustrating', 'annoying', 'bothersome'],
            'calm': ['peaceful', 'tranquil', 'serene', 'calm', 'gentle']
        }
        
        # 情绪缓冲词汇
        self.buffer_phrases = {
            'intense': ['请注意，以下内容可能较为强烈', '提醒：内容可能引起强烈情绪反应'],
            'moderate': ['以下内容可能引起一些情绪反应', '请注意：内容可能令人不适'],
            'mild': ['以下内容可能引起轻微情绪反应', '提醒：内容可能令人不快'],
            'calm': ['以下内容相对平静', '内容较为温和']
        }
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for layer in self.emotional_intensity_assessor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        for layer in self.emotion_buffer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, content_embedding: torch.Tensor, text: str) -> Tuple[str, Dict[str, Any]]:
        """
        应用情绪缓冲
        
        Args:
            content_embedding: 内容嵌入 [batch_size, hidden_size]
            text: 原始文本
            
        Returns:
            缓冲后的文本
            缓冲分析信息
        """
        # 评估情绪强度
        emotional_intensity = self.emotional_intensity_assessor(content_embedding).item()
        
        # 确定情绪等级
        emotion_level = self._determine_emotion_level(emotional_intensity)
        
        # 应用情绪缓冲
        buffered_text = self._apply_emotion_buffer(text, emotion_level)
        
        # 缓冲分析
        buffer_analysis = {
            'emotional_intensity': emotional_intensity,
            'emotion_level': emotion_level.value,
            'buffer_applied': emotion_level != EmotionalIntensity.CALM,
            'buffer_phrases_added': len(self._get_buffer_phrases(emotion_level)) > 0
        }
        
        return buffered_text, buffer_analysis
    
    def _determine_emotion_level(self, intensity: float) -> EmotionalIntensity:
        """确定情绪等级"""
        if intensity < 0.2:
            return EmotionalIntensity.CALM
        elif intensity < 0.4:
            return EmotionalIntensity.MILD
        elif intensity < 0.6:
            return EmotionalIntensity.MODERATE
        elif intensity < 0.8:
            return EmotionalIntensity.INTENSE
        else:
            return EmotionalIntensity.OVERWHELMING
    
    def _apply_emotion_buffer(self, text: str, emotion_level: EmotionalIntensity) -> str:
        """应用情绪缓冲"""
        if emotion_level == EmotionalIntensity.CALM:
            return text
        
        # 获取缓冲短语
        buffer_phrases = self._get_buffer_phrases(emotion_level)
        
        # 添加缓冲短语
        if buffer_phrases:
            buffer_text = buffer_phrases[0] + "\n\n"
            buffered_text = buffer_text + text
        else:
            buffered_text = text
        
        return buffered_text
    
    def _get_buffer_phrases(self, emotion_level: EmotionalIntensity) -> List[str]:
        """获取缓冲短语"""
        level_key = emotion_level.value
        return self.buffer_phrases.get(level_key, [])


class CognitiveVaccine:
    """V4.0 认知疫苗机制"""
    
    def __init__(self, hidden_size: int = 768):
        self.hidden_size = hidden_size
        
        # 初始化认知降维包和情绪缓冲结构
        self.cognitive_downgrade = CognitiveDowngrade(hidden_size)
        self.emotion_buffer = EmotionBuffer(hidden_size)
        
        # 疫苗配置
        self.default_cognitive_level = CognitiveLevel.ADULT
        self.enable_emotion_buffer = True
        
        # 疫苗应用历史
        self.vaccine_history = []
    
    def apply_vaccine(self, content_embedding: torch.Tensor, text: str, 
                     target_cognitive_level: CognitiveLevel = None,
                     enable_emotion_buffer: bool = None) -> VaccinatedContent:
        """
        应用认知疫苗
        
        Args:
            content_embedding: 内容嵌入 [batch_size, hidden_size]
            text: 原始文本
            target_cognitive_level: 目标认知等级
            enable_emotion_buffer: 是否启用情绪缓冲
            
        Returns:
            接种疫苗后的内容
        """
        # 使用默认配置
        if target_cognitive_level is None:
            target_cognitive_level = self.default_cognitive_level
        if enable_emotion_buffer is None:
            enable_emotion_buffer = self.enable_emotion_buffer
        
        # 应用认知降维包
        downgraded_text, downgrade_analysis = self.cognitive_downgrade(
            content_embedding, text, target_cognitive_level
        )
        
        # 应用情绪缓冲结构
        if enable_emotion_buffer:
            buffered_text, buffer_analysis = self.emotion_buffer(content_embedding, downgraded_text)
        else:
            buffered_text = downgraded_text
            buffer_analysis = {'buffer_applied': False}
        
        # 确定最终认知等级和情绪强度
        final_cognitive_level = self._determine_final_cognitive_level(downgrade_analysis)
        final_emotional_intensity = self._determine_final_emotional_intensity(buffer_analysis)
        
        # 记录疫苗应用
        vaccine_applied = []
        if downgrade_analysis.get('vocabulary_simplified'):
            vaccine_applied.append('vocabulary_simplification')
        if downgrade_analysis.get('sentences_simplified'):
            vaccine_applied.append('sentence_simplification')
        if downgrade_analysis.get('structure_simplified'):
            vaccine_applied.append('structure_simplification')
        if buffer_analysis.get('buffer_applied'):
            vaccine_applied.append('emotion_buffer')
        
        # 计算置信度
        confidence_score = self._calculate_vaccine_confidence(downgrade_analysis, buffer_analysis)
        
        # 创建接种疫苗后的内容
        vaccinated_content = VaccinatedContent(
            original_content=text,
            vaccinated_content=buffered_text,
            cognitive_level=final_cognitive_level,
            emotional_intensity=final_emotional_intensity,
            vaccine_applied=vaccine_applied,
            confidence_score=confidence_score
        )
        
        # 记录疫苗历史
        self._record_vaccine_application(vaccinated_content)
        
        return vaccinated_content
    
    def _determine_final_cognitive_level(self, downgrade_analysis: Dict[str, Any]) -> CognitiveLevel:
        """确定最终认知等级"""
        target_level = downgrade_analysis.get('target_level', 'adult')
        
        # 根据降维强度调整认知等级
        downgrade_intensity = downgrade_analysis.get('downgrade_intensity', 0.3)
        
        if downgrade_intensity > 0.7:
            return CognitiveLevel.CHILD
        elif downgrade_intensity > 0.5:
            return CognitiveLevel.TEEN
        elif downgrade_intensity > 0.3:
            return CognitiveLevel.ADULT
        else:
            return CognitiveLevel.EXPERT
    
    def _determine_final_emotional_intensity(self, buffer_analysis: Dict[str, Any]) -> EmotionalIntensity:
        """确定最终情绪强度"""
        if buffer_analysis.get('buffer_applied', False):
            emotion_level = buffer_analysis.get('emotion_level', 'calm')
            return EmotionalIntensity(emotion_level)
        else:
            return EmotionalIntensity.CALM
    
    def _calculate_vaccine_confidence(self, downgrade_analysis: Dict[str, Any], 
                                    buffer_analysis: Dict[str, Any]) -> float:
        """计算疫苗置信度"""
        # 基于疫苗应用效果计算置信度
        vaccine_effects = 0
        
        if downgrade_analysis.get('vocabulary_simplified'):
            vaccine_effects += 0.3
        if downgrade_analysis.get('sentences_simplified'):
            vaccine_effects += 0.3
        if downgrade_analysis.get('structure_simplified'):
            vaccine_effects += 0.2
        if buffer_analysis.get('buffer_applied'):
            vaccine_effects += 0.2
        
        return min(vaccine_effects, 1.0)
    
    def _record_vaccine_application(self, vaccinated_content: VaccinatedContent):
        """记录疫苗应用历史"""
        record = {
            'timestamp': time.time(),
            'original_length': len(vaccinated_content.original_content),
            'vaccinated_length': len(vaccinated_content.vaccinated_content),
            'cognitive_level': vaccinated_content.cognitive_level.value,
            'emotional_intensity': vaccinated_content.emotional_intensity.value,
            'vaccine_applied': vaccinated_content.vaccine_applied,
            'confidence_score': vaccinated_content.confidence_score
        }
        
        self.vaccine_history.append(record)
        
        # 保持历史记录在合理范围内
        if len(self.vaccine_history) > 1000:
            self.vaccine_history = self.vaccine_history[-500:]
    
    def get_vaccine_statistics(self) -> Dict[str, Any]:
        """获取疫苗应用统计信息"""
        if not self.vaccine_history:
            return {"error": "No vaccine history available"}
        
        total_applications = len(self.vaccine_history)
        
        # 认知等级分布
        cognitive_levels = [record['cognitive_level'] for record in self.vaccine_history]
        cognitive_distribution = {}
        for level in cognitive_levels:
            cognitive_distribution[level] = cognitive_distribution.get(level, 0) + 1
        
        # 情绪强度分布
        emotional_intensities = [record['emotional_intensity'] for record in self.vaccine_history]
        emotional_distribution = {}
        for intensity in emotional_intensities:
            emotional_distribution[intensity] = emotional_distribution.get(intensity, 0) + 1
        
        # 疫苗应用效果统计
        vaccine_effects = []
        for record in self.vaccine_history:
            if record['vaccine_applied']:
                vaccine_effects.append(len(record['vaccine_applied']))
        
        avg_vaccine_effects = sum(vaccine_effects) / len(vaccine_effects) if vaccine_effects else 0
        
        return {
            'total_applications': total_applications,
            'cognitive_level_distribution': cognitive_distribution,
            'emotional_intensity_distribution': emotional_distribution,
            'average_vaccine_effects': avg_vaccine_effects,
            'average_confidence_score': sum(record['confidence_score'] for record in self.vaccine_history) / total_applications
        }


def test_cognitive_vaccine():
    """测试认知疫苗机制"""
    print("🧪 测试V4.0认知疫苗机制...")
    
    # 创建认知疫苗
    vaccine = CognitiveVaccine()
    
    # 测试用例
    test_cases = [
        {
            'text': '这是一个简单的测试文本。',
            'target_cognitive_level': CognitiveLevel.ADULT,
            'expected_result': 'simple'
        },
        {
            'text': '这是一个包含复杂词汇和句式的文本，需要深入的理论分析和系统性的方法论框架来理解其内涵。',
            'target_cognitive_level': CognitiveLevel.CHILD,
            'expected_result': 'simplified'
        },
        {
            'text': '这是一个可能引起强烈情绪反应的内容，包含令人震惊和不安的信息。',
            'target_cognitive_level': CognitiveLevel.ADULT,
            'expected_result': 'buffered'
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n测试用例 {i+1}:")
        print(f"原始文本: {test_case['text']}")
        print(f"目标认知等级: {test_case['target_cognitive_level'].value}")
        
        # 创建模拟嵌入
        content_embedding = torch.randn(1, 768)
        
        # 应用认知疫苗
        vaccinated_content = vaccine.apply_vaccine(
            content_embedding,
            test_case['text'],
            test_case['target_cognitive_level']
        )
        
        print(f"接种疫苗后文本: {vaccinated_content.vaccinated_content}")
        print(f"认知等级: {vaccinated_content.cognitive_level.value}")
        print(f"情绪强度: {vaccinated_content.emotional_intensity.value}")
        print(f"疫苗应用: {vaccinated_content.vaccine_applied}")
        print(f"置信度: {vaccinated_content.confidence_score:.3f}")
    
    # 获取统计信息
    stats = vaccine.get_vaccine_statistics()
    print(f"\n📊 疫苗应用统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_cognitive_vaccine() 