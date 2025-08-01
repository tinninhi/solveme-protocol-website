#!/usr/bin/env python3
"""
增强安全系统 - Enhanced Safety System
完善表达裁决器、认知疫苗和安全指数，确保AI的负责任发展
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class SafetyLevel(Enum):
    """安全等级"""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    DANGEROUS = "dangerous"

class ContentType(Enum):
    """内容类型"""
    GENERAL = "general"
    TECHNICAL = "technical"
    EMOTIONAL = "emotional"
    SENSITIVE = "sensitive"
    HARMFUL = "harmful"

@dataclass
class SafetyAssessment:
    """安全评估结果"""
    safety_level: SafetyLevel
    safety_score: float
    risk_factors: List[str]
    recommendations: List[str]
    content_type: ContentType
    cognitive_complexity: float
    emotional_impact: float

class EnhancedExpressionArbitrator(nn.Module):
    """增强表达裁决器"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # S-Index 安全指数增强
        self.safety_index = EnhancedSafetyIndex(hidden_size)
        
        # H-Index 人类接收指数增强
        self.human_index = EnhancedHumanIndex(hidden_size)
        
        # L-Index 语言复杂度指数增强
        self.language_index = EnhancedLanguageIndex(hidden_size)
        
        # 综合评估器
        self.comprehensive_assessor = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 5),  # 5个评估维度
            nn.Sigmoid()
        )
        
        # 内容分类器
        self.content_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, len(ContentType)),
            nn.Softmax(dim=-1)
        )
        
        # 风险检测器
        self.risk_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 10),  # 10种风险类型
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.comprehensive_assessor, self.content_classifier, self.risk_detector]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, content_embedding: torch.Tensor, text: str) -> Tuple[SafetyAssessment, Dict[str, Any]]:
        """
        增强表达裁决
        
        Args:
            content_embedding: 内容嵌入 [batch_size, hidden_size]
            text: 原始文本
            
        Returns:
            安全评估结果
            详细分析信息
        """
        # S-Index 安全评估
        safety_score, safety_analysis = self.safety_index(content_embedding, text)
        
        # H-Index 人类接收评估
        human_score, human_analysis = self.human_index(content_embedding, text)
        
        # L-Index 语言复杂度评估
        language_score, language_analysis = self.language_index(content_embedding, text)
        
        # 综合评估
        combined_features = torch.cat([
            safety_analysis['features'],
            human_analysis['features'],
            language_analysis['features']
        ], dim=-1)
        
        comprehensive_scores = self.comprehensive_assessor(combined_features)
        
        # 内容分类
        content_probs = self.content_classifier(content_embedding.mean(dim=1))
        content_type_idx = torch.argmax(content_probs, dim=-1).item()
        content_type = list(ContentType)[content_type_idx]
        
        # 风险检测
        risk_scores = self.risk_detector(content_embedding.mean(dim=1))
        risk_factors = self._identify_risk_factors(risk_scores, text)
        
        # 确定安全等级
        overall_score = (safety_score + human_score + language_score) / 3
        safety_level = self._determine_safety_level(overall_score, risk_factors)
        
        # 生成建议
        recommendations = self._generate_recommendations(
            safety_level, safety_score, human_score, language_score, risk_factors
        )
        
        # 创建安全评估结果
        assessment = SafetyAssessment(
            safety_level=safety_level,
            safety_score=overall_score,
            risk_factors=risk_factors,
            recommendations=recommendations,
            content_type=content_type,
            cognitive_complexity=language_score,
            emotional_impact=human_score
        )
        
        # 详细分析信息
        detailed_analysis = {
            'safety_analysis': safety_analysis,
            'human_analysis': human_analysis,
            'language_analysis': language_analysis,
            'comprehensive_scores': comprehensive_scores.detach().cpu().numpy().tolist(),
            'content_type_probs': content_probs.detach().cpu().numpy().tolist(),
            'risk_scores': risk_scores.detach().cpu().numpy().tolist(),
            'overall_score': overall_score
        }
        
        return assessment, detailed_analysis
    
    def _identify_risk_factors(self, risk_scores: torch.Tensor, text: str) -> List[str]:
        """识别风险因素"""
        risk_types = [
            "harmful_content", "bias", "misinformation", "privacy_violation",
            "emotional_manipulation", "cognitive_overload", "complex_terminology",
            "sensitive_topics", "inappropriate_humor", "conflicting_advice"
        ]
        
        risk_factors = []
        for i, risk_type in enumerate(risk_types):
            if risk_scores[0, i].item() > 0.5:
                risk_factors.append(risk_type)
        
        # 基于文本内容的额外检查
        if any(word in text.lower() for word in ['harm', 'danger', 'risk']):
            risk_factors.append("explicit_risk_mention")
        
        return risk_factors
    
    def _determine_safety_level(self, overall_score: float, risk_factors: List[str]) -> SafetyLevel:
        """确定安全等级"""
        if overall_score >= 0.8 and not risk_factors:
            return SafetyLevel.SAFE
        elif overall_score >= 0.6 and len(risk_factors) <= 2:
            return SafetyLevel.CAUTION
        elif overall_score >= 0.4 and len(risk_factors) <= 4:
            return SafetyLevel.WARNING
        else:
            return SafetyLevel.DANGEROUS
    
    def _generate_recommendations(self, safety_level: SafetyLevel, safety_score: float, 
                                 human_score: float, language_score: float, 
                                 risk_factors: List[str]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if safety_level == SafetyLevel.DANGEROUS:
            recommendations.append("立即停止生成，内容存在严重安全风险")
            recommendations.append("重新评估输入和上下文")
        
        if safety_score < 0.7:
            recommendations.append("提高内容安全性，避免有害信息")
        
        if human_score < 0.6:
            recommendations.append("简化表达，降低认知负荷")
            recommendations.append("考虑用户情绪状态")
        
        if language_score > 0.8:
            recommendations.append("降低语言复杂度，使用更简单的表达")
        
        if "harmful_content" in risk_factors:
            recommendations.append("移除或重新表述有害内容")
        
        if "cognitive_overload" in risk_factors:
            recommendations.append("分段表达，降低信息密度")
        
        return recommendations

class EnhancedSafetyIndex(nn.Module):
    """增强安全指数"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 有害内容检测器
        self.harmful_content_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 偏见检测器
        self.bias_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 误导信息检测器
        self.misinformation_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 隐私保护检测器
        self.privacy_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.harmful_content_detector, self.bias_detector, 
                      self.misinformation_detector, self.privacy_detector, self.feature_extractor]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, content_embedding: torch.Tensor, text: str) -> Tuple[float, Dict[str, Any]]:
        """增强安全评估"""
        # 提取特征
        features = self.feature_extractor(content_embedding.mean(dim=1))
        
        # 各项安全检测
        harmful_score = self.harmful_content_detector(features)
        bias_score = self.bias_detector(features)
        misinformation_score = self.misinformation_detector(features)
        privacy_score = self.privacy_detector(features)
        
        # 综合安全分数
        safety_score = 1.0 - (harmful_score + bias_score + misinformation_score + privacy_score) / 4
        
        # 详细分析
        analysis = {
            'features': features,
            'harmful_score': harmful_score.item(),
            'bias_score': bias_score.item(),
            'misinformation_score': misinformation_score.item(),
            'privacy_score': privacy_score.item(),
            'safety_score': safety_score.item()
        }
        
        return safety_score.item(), analysis

class EnhancedHumanIndex(nn.Module):
    """增强人类接收指数"""
    
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
        
        # 情绪影响评估器
        self.emotional_impact_assessor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 文化适应性评估器
        self.cultural_adaptability_assessor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.cognitive_load_assessor, self.emotional_impact_assessor,
                      self.cultural_adaptability_assessor, self.feature_extractor]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, content_embedding: torch.Tensor, text: str) -> Tuple[float, Dict[str, Any]]:
        """增强人类接收评估"""
        # 提取特征
        features = self.feature_extractor(content_embedding.mean(dim=1))
        
        # 各项评估
        cognitive_load = self.cognitive_load_assessor(features)
        emotional_impact = self.emotional_impact_assessor(features)
        cultural_adaptability = self.cultural_adaptability_assessor(features)
        
        # 综合人类接收分数
        human_score = (1.0 - cognitive_load + emotional_impact + cultural_adaptability) / 3
        
        # 详细分析
        analysis = {
            'features': features,
            'cognitive_load': cognitive_load.item(),
            'emotional_impact': emotional_impact.item(),
            'cultural_adaptability': cultural_adaptability.item(),
            'human_score': human_score.item()
        }
        
        return human_score.item(), analysis

class EnhancedLanguageIndex(nn.Module):
    """增强语言复杂度指数"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 词汇复杂度评估器
        self.vocabulary_complexity_assessor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 句式复杂度评估器
        self.sentence_complexity_assessor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 概念复杂度评估器
        self.concept_complexity_assessor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.vocabulary_complexity_assessor, self.sentence_complexity_assessor,
                      self.concept_complexity_assessor, self.feature_extractor]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, content_embedding: torch.Tensor, text: str) -> Tuple[float, Dict[str, Any]]:
        """增强语言复杂度评估"""
        # 提取特征
        features = self.feature_extractor(content_embedding.mean(dim=1))
        
        # 各项复杂度评估
        vocabulary_complexity = self.vocabulary_complexity_assessor(features)
        sentence_complexity = self.sentence_complexity_assessor(features)
        concept_complexity = self.concept_complexity_assessor(features)
        
        # 综合语言复杂度分数（越低越好）
        language_score = 1.0 - (vocabulary_complexity + sentence_complexity + concept_complexity) / 3
        
        # 详细分析
        analysis = {
            'features': features,
            'vocabulary_complexity': vocabulary_complexity.item(),
            'sentence_complexity': sentence_complexity.item(),
            'concept_complexity': concept_complexity.item(),
            'language_score': language_score.item()
        }
        
        return language_score.item(), analysis

class EnhancedCognitiveVaccine(nn.Module):
    """增强认知疫苗机制"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 认知降维包增强
        self.cognitive_downgrade = EnhancedCognitiveDowngrade(hidden_size)
        
        # 情绪缓冲结构增强
        self.emotion_buffer = EnhancedEmotionBuffer(hidden_size)
        
        # 安全过滤器
        self.safety_filter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for layer in self.safety_filter:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, content_embedding: torch.Tensor, text: str, 
                target_level: str = "adult") -> Tuple[str, Dict[str, Any]]:
        """增强认知疫苗应用"""
        # 安全检查
        safety_score = self.safety_filter(content_embedding.mean(dim=1))
        
        # 如果内容不安全，直接拒绝
        if safety_score.item() < 0.3:
            return "内容存在安全风险，已拒绝生成。", {
                'safety_score': safety_score.item(),
                'action': 'rejected',
                'reason': 'safety_risk'
            }
        
        # 认知降维
        downgraded_text, downgrade_analysis = self.cognitive_downgrade(
            content_embedding, text, target_level
        )
        
        # 情绪缓冲
        buffered_text, buffer_analysis = self.emotion_buffer(
            content_embedding, downgraded_text
        )
        
        # 综合分析
        analysis = {
            'safety_score': safety_score.item(),
            'downgrade_analysis': downgrade_analysis,
            'buffer_analysis': buffer_analysis,
            'action': 'processed',
            'target_level': target_level
        }
        
        return buffered_text, analysis

class EnhancedCognitiveDowngrade(nn.Module):
    """增强认知降维包"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 复杂度评估器
        self.complexity_assessor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 降维强度控制器
        self.downgrade_controller = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 词汇简化映射
        self.vocabulary_simplification = {
            "sophisticated": "advanced",
            "elaborate": "detailed",
            "comprehensive": "complete",
            "theoretical": "concept",
            "empirical": "practical",
            "methodological": "method",
            "analytical": "analysis",
            "systematic": "organized",
            "paradigm": "model",
            "framework": "structure",
            "algorithm": "method",
            "optimization": "improvement",
            "implementation": "use",
            "architecture": "design",
            "protocol": "rule"
        }
        
        # 句式简化模式
        self.sentence_simplification = {
            r"notwithstanding the fact that": "although",
            r"in light of the aforementioned": "given this",
            r"it is imperative to note that": "it's important that",
            r"furthermore, it should be emphasized": "also,",
            r"consequently, it follows that": "so,",
            r"in accordance with": "following",
            r"with respect to": "about",
            r"in terms of": "for"
        }
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.complexity_assessor, self.downgrade_controller]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, content_embedding: torch.Tensor, text: str, 
                target_level: str = "adult") -> Tuple[str, Dict[str, Any]]:
        """增强认知降维"""
        # 评估原始复杂度
        original_complexity = self.complexity_assessor(content_embedding.mean(dim=1)).item()
        
        # 确定降维强度
        downgrade_intensity = self.downgrade_controller(content_embedding.mean(dim=1)).item()
        
        # 根据目标等级调整强度
        level_intensity = self._get_level_intensity(target_level)
        final_intensity = min(downgrade_intensity, level_intensity)
        
        # 应用词汇简化
        simplified_text = self._simplify_vocabulary(text, final_intensity)
        
        # 应用句式简化
        simplified_text = self._simplify_sentences(simplified_text, final_intensity)
        
        # 应用结构简化
        simplified_text = self._simplify_structure(simplified_text, final_intensity)
        
        # 分析结果
        analysis = {
            'original_complexity': original_complexity,
            'downgrade_intensity': final_intensity,
            'target_level': target_level,
            'vocabulary_simplified': len(self._get_simplified_words(text)) > 0,
            'sentences_simplified': len(self._get_simplified_sentences(text)) > 0,
            'structure_simplified': len(self._get_simplified_structures(text)) > 0
        }
        
        return simplified_text, analysis
    
    def _get_level_intensity(self, target_level: str) -> float:
        """获取等级强度"""
        intensity_mapping = {
            "infant": 0.9,
            "child": 0.7,
            "teen": 0.5,
            "adult": 0.3,
            "expert": 0.1
        }
        return intensity_mapping.get(target_level, 0.3)
    
    def _simplify_vocabulary(self, text: str, intensity: float) -> str:
        """简化词汇"""
        simplified_text = text
        
        # 根据强度应用简化
        for complex_word, simple_word in self.vocabulary_simplification.items():
            if np.random.random() < intensity:
                pattern = r'\b' + re.escape(complex_word) + r'\b'
                simplified_text = re.sub(pattern, simple_word, simplified_text, flags=re.IGNORECASE)
        
        return simplified_text
    
    def _simplify_sentences(self, text: str, intensity: float) -> str:
        """简化句式"""
        simplified_text = text
        
        # 根据强度应用句式简化
        for pattern, replacement in self.sentence_simplification.items():
            if np.random.random() < intensity:
                simplified_text = re.sub(pattern, replacement, simplified_text, flags=re.IGNORECASE)
        
        return simplified_text
    
    def _simplify_structure(self, text: str, intensity: float) -> str:
        """简化结构"""
        if intensity > 0.5:
            # 分割长句
            sentences = re.split(r'[.!?]+', text)
            simplified_sentences = []
            
            for sentence in sentences:
                if len(sentence.split()) > 20:  # 长句分割
                    words = sentence.split()
                    mid = len(words) // 2
                    simplified_sentences.append(' '.join(words[:mid]) + '.')
                    simplified_sentences.append(' '.join(words[mid:]) + '.')
                else:
                    simplified_sentences.append(sentence)
            
            return ' '.join(simplified_sentences)
        
        return text
    
    def _get_simplified_words(self, text: str) -> List[str]:
        """获取简化的词汇"""
        simplified = []
        for complex_word in self.vocabulary_simplification.keys():
            if complex_word.lower() in text.lower():
                simplified.append(complex_word)
        return simplified
    
    def _get_simplified_sentences(self, text: str) -> List[str]:
        """获取简化的句式"""
        simplified = []
        for pattern in self.sentence_simplification.keys():
            if re.search(pattern, text, re.IGNORECASE):
                simplified.append(pattern)
        return simplified
    
    def _get_simplified_structures(self, text: str) -> List[str]:
        """获取简化的结构"""
        # 检测长句
        sentences = re.split(r'[.!?]+', text)
        long_sentences = [s for s in sentences if len(s.split()) > 20]
        return long_sentences

class EnhancedEmotionBuffer(nn.Module):
    """增强情绪缓冲结构"""
    
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 情绪强度评估器
        self.emotion_intensity_assessor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 情绪类型分类器
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 7),  # 7种基本情绪
            nn.Softmax(dim=-1)
        )
        
        # 缓冲强度控制器
        self.buffer_controller = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # 情绪词汇映射
        self.emotion_softening = {
            "terrible": "difficult",
            "horrible": "challenging",
            "awful": "tough",
            "amazing": "good",
            "incredible": "impressive",
            "fantastic": "great",
            "wonderful": "nice"
        }
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.emotion_intensity_assessor, self.emotion_classifier, self.buffer_controller]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, content_embedding: torch.Tensor, text: str) -> Tuple[str, Dict[str, Any]]:
        """增强情绪缓冲"""
        # 评估情绪强度
        emotion_intensity = self.emotion_intensity_assessor(content_embedding.mean(dim=1)).item()
        
        # 分类情绪类型
        emotion_probs = self.emotion_classifier(content_embedding.mean(dim=1))
        emotion_type_idx = torch.argmax(emotion_probs, dim=-1).item()
        emotion_types = ["neutral", "joy", "sadness", "anger", "fear", "surprise", "disgust"]
        detected_emotion = emotion_types[emotion_type_idx]
        
        # 确定缓冲强度
        buffer_intensity = self.buffer_controller(content_embedding.mean(dim=1)).item()
        
        # 应用情绪缓冲
        buffered_text = self._apply_emotion_buffer(text, emotion_intensity, buffer_intensity)
        
        # 分析结果
        analysis = {
            'emotion_intensity': emotion_intensity,
            'detected_emotion': detected_emotion,
            'emotion_probs': emotion_probs.detach().cpu().numpy().tolist(),
            'buffer_intensity': buffer_intensity,
            'emotion_softened': len(self._get_softened_emotions(text)) > 0
        }
        
        return buffered_text, analysis
    
    def _apply_emotion_buffer(self, text: str, intensity: float, buffer_strength: float) -> str:
        """应用情绪缓冲"""
        if intensity < 0.3:  # 情绪强度低，无需缓冲
            return text
        
        buffered_text = text
        
        # 根据缓冲强度应用情绪软化
        for strong_emotion, soft_emotion in self.emotion_softening.items():
            if np.random.random() < buffer_strength:
                pattern = r'\b' + re.escape(strong_emotion) + r'\b'
                buffered_text = re.sub(pattern, soft_emotion, buffered_text, flags=re.IGNORECASE)
        
        # 添加情绪缓冲前缀
        if intensity > 0.7 and buffer_strength > 0.5:
            buffer_prefixes = [
                "请保持冷静，",
                "让我们理性地看待，",
                "从客观角度来说，",
                "需要说明的是，"
            ]
            buffered_text = np.random.choice(buffer_prefixes) + buffered_text
        
        return buffered_text
    
    def _get_softened_emotions(self, text: str) -> List[str]:
        """获取软化的情绪词汇"""
        softened = []
        for strong_emotion in self.emotion_softening.keys():
            if strong_emotion.lower() in text.lower():
                softened.append(strong_emotion)
        return softened

def main():
    """主函数"""
    print("🛡️ 增强安全系统")
    print("=" * 60)
    
    # 创建增强安全系统
    safety_system = EnhancedExpressionArbitrator(hidden_size=768)
    cognitive_vaccine = EnhancedCognitiveVaccine(hidden_size=768)
    
    # 测试示例
    test_text = "这是一个非常复杂的技术问题，需要深入的理论分析和实证研究。"
    test_embedding = torch.randn(1, 10, 768)  # 模拟嵌入
    
    print("🧪 测试增强安全系统...")
    
    # 测试表达裁决器
    assessment, analysis = safety_system(test_embedding, test_text)
    print(f"安全评估结果:")
    print(f"  安全等级: {assessment.safety_level.value}")
    print(f"  安全分数: {assessment.safety_score:.3f}")
    print(f"  风险因素: {assessment.risk_factors}")
    print(f"  建议: {assessment.recommendations}")
    
    # 测试认知疫苗
    vaccinated_text, vaccine_analysis = cognitive_vaccine(test_embedding, test_text, "adult")
    print(f"\n认知疫苗结果:")
    print(f"  原始文本: {test_text}")
    print(f"  处理后: {vaccinated_text}")
    print(f"  处理分析: {vaccine_analysis}")
    
    print(f"\n✅ 增强安全系统已就绪！")
    print("=" * 60)

if __name__ == "__main__":
    main() 