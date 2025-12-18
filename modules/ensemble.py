# modules/ensemble.py
import numpy as np
import joblib
import os

class SpikeHunterEnsemble:
    """
    LightGBM과 XGBoost 모델을 결합한 소프트 보팅 앙상블 래퍼 클래스.
    단일 모델과 동일한 인터페이스(predict_proba, feature_names_in_)를 제공하여
    기존 파이프라인과 호환성을 유지합니다.
    """
    def __init__(self, models: list, weights: list = None):
        self.models = models
        self.weights = weights if weights else [1.0] * len(models)
        # 가중치 정규화
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        # 피처 이름은 첫 번째 모델(LightGBM) 기준
        self.feature_names_in_ = models[0].feature_names_in_ if hasattr(models[0], 'feature_names_in_') else []

    def predict_proba(self, X):
        """앙상블 예측 확률 반환 (Soft Voting)"""
        # XGBoost는 feature 순서에 민감하므로 컬럼 순서 강제 정렬
        if hasattr(X, 'columns'):
            X = X[self.feature_names_in_]
            
        probas = []
        for model in self.models:
            # 각 모델별 예측 확률 계산
            p = model.predict_proba(X)
            probas.append(p)
            
        # 가중 평균 계산
        final_proba = np.zeros_like(probas[0])
        for p, w in zip(probas, self.weights):
            final_proba += p * w
            
        return final_proba

    def predict(self, X):
        """예측 클래스 반환 (기본 임계값 0.5)"""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def save(self, path):
        joblib.dump(self, path)
        
    @staticmethod
    def load(path):
        return joblib.load(path)