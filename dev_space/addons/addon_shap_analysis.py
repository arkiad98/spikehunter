# addons/addon_shap_analysis.py (ver 2.7.12)
"""
SHAP(SHapley Additive exPlanations) 라이브러리를 사용하여
ML 모델의 예측 결과를 해석하고 설명하는 분석 애드온 모듈.

[v2.7.12] 회귀 모델 분석 및 타임스탬프 파일명 적용
- 회귀 모델(target_model.joblib)에 대한 SHAP 분석 기능 추가.
- 모든 SHAP 리포트 이미지 파일명에 타임스탬프를 추가하여 분석 이력 관리.
"""
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
import optuna

from modules.utils_io import read_yaml, ensure_dir, get_stock_names, load_partition_day
from modules.utils_logger import logger
from modules import utils_db

def _set_korean_font():
    """matplotlib에서 한글 폰트를 설정하여 그래프의 가독성을 높입니다."""
    font_path = 'c:/Windows/Fonts/malgun.ttf' # Windows 기준
    if not os.path.exists(font_path):
        font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
        
    if os.path.exists(font_path):
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
    else:
        logger.warning("한글 폰트(맑은 고딕 또는 나눔고딕)를 찾을 수 없어 기본 폰트로 출력됩니다.")
    plt.rcParams['axes.unicode_minus'] = False

def _get_latest_prediction_file(pred_path: str):
    """predictions 폴더에서 가장 최신 추천 종목 CSV 파일을 찾습니다."""
    if not os.path.exists(pred_path): return None
    files = [os.path.join(pred_path, f) for f in os.listdir(pred_path) if f.endswith('.csv') and '_targets' not in f]
    if not files: return None
    return max(files, key=os.path.getctime)

def run_optimization_shap_analysis(study: optuna.study.Study, output_dir: str):
    """
    완료된 Optuna Study 객체를 분석하여 하이퍼파라미터의 중요도를 시각화하고 DB에 기록합니다.
    """
    logger.info("최적화 결과에 대한 SHAP 분석을 시작합니다...")
    
    df = study.trials_dataframe()
    df = df[df['state'] == 'COMPLETE']
    
    if len(df) < 10:
        logger.warning(f"분석할 Trial의 수가 너무 적어({len(df)}개) SHAP 분석을 건너뜁니다.")
        return

    param_cols = [col for col in df.columns if col.startswith('params_')]
    X = df[param_cols]
    X.columns = [col.replace('params_', '') for col in X.columns]
    y = df['value']

    surrogate_model = lgb.LGBMRegressor(random_state=42, n_jobs=1)
    surrogate_model.fit(X, y)

    explainer = shap.TreeExplainer(surrogate_model)
    shap_values = explainer.shap_values(X)

    shap_df = pd.DataFrame({
        'item_name': X.columns,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    })
    shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)
    shap_df['rank'] = range(1, len(shap_df) + 1)
    
    utils_db.insert_shap_results(
        run_id=study.study_name,
        analysis_type='optimization_hyperparameter',
        shap_df=shap_df
    )
    
    _set_korean_font()
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    # [수정] Study 이름(최적화 내용 + 타임스탬프)을 제목에 동적으로 추가
    title = f"Hyperparameter Importance (SHAP)\nStudy: {study.study_name}"
    plt.title(title, fontsize=12)
    plt.xlabel("평균 SHAP 값 (성과에 대한 기여도)", fontsize=12)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "optimization_shap_summary.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"최적화 파라미터 SHAP 분석 결과가 '{output_path}'에 저장되었습니다.")


def run_shap_analysis(settings_path: str, analysis_type: str = 'global_classification'):
    """
    SHAP 분석을 실행하고 결과를 시각화 및 DB에 저장합니다.
    """
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger.info("="*60)
    logger.info(f"      <<< SHAP 모델 해석 ({analysis_type}) 시작 >>>")
    logger.info("="*60)
    
    cfg = read_yaml(settings_path)
    paths = cfg["paths"]

    # 1. 분석 유형에 따라 모델, 데이터셋, 타겟 컬럼 설정
    if analysis_type == 'global_classification':
        model_filename = "lgbm_model.joblib"
        dataset_filename = "ml_classification_dataset.parquet"
        target_col = 'target'
        title = "분류 모델 SHAP 전역 피처 중요도"
    elif analysis_type == 'global_regression':
        model_filename = "target_model.joblib"
        dataset_filename = "ml_regression_dataset.parquet"
        target_col = 'target_max_ret'
        title = "회귀 모델 SHAP 전역 피처 중요도"
    elif analysis_type == 'local':
        model_filename = "lgbm_model.joblib" # Local 분석은 분류 모델 기준
    else:
        logger.error(f"지원하지 않는 분석 유형입니다: {analysis_type}")
        return

    # 2. 모델 로드
    model_path = os.path.join(paths["models"], model_filename)
    if not os.path.exists(model_path):
        logger.error(f"학습된 ML 모델({model_path})이 없습니다. 모델 학습을 먼저 실행해주세요.")
        return
    model = joblib.load(model_path)
    logger.info(f"ML 모델 로드 완료: {model_path}")

    # 3. SHAP Explainer 생성
    logger.info("SHAP TreeExplainer를 생성합니다...")
    explainer = shap.TreeExplainer(model)

    # 4. 전역 분석 (Global Analysis)
    if analysis_type.startswith('global'):
        dataset_path = os.path.join(paths["ml_dataset"], dataset_filename)
        if not os.path.exists(dataset_path):
            logger.error(f"ML 데이터셋({dataset_path})이 없습니다. 데이터셋 생성을 먼저 실행해주세요.")
            return
        df = pd.read_parquet(dataset_path)
        feature_cols = [col for col in df.columns if col not in ['date', 'code', target_col]]
        X, y = df[feature_cols], df[target_col]
        from sklearn.model_selection import train_test_split
        _, X_test, _, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

        logger.info("전체 테스트 데이터셋에 대한 SHAP 값을 계산합니다...")
        shap_values = explainer.shap_values(X_test)
        logger.info("SHAP 값 계산 완료.")
        
        shap_values_for_calc = shap_values[1] if isinstance(shap_values, list) else shap_values
        
        shap_df = pd.DataFrame({
            'item_name': X_test.columns,
            'mean_abs_shap': np.abs(shap_values_for_calc).mean(axis=0)
        })
        shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)
        shap_df['rank'] = range(1, len(shap_df) + 1)

        utils_db.insert_shap_results(
            run_id=f"model_train_{run_timestamp}",
            analysis_type=analysis_type,
            shap_df=shap_df
        )

        _set_korean_font()
        plt.figure()
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        plt.title(title, fontsize=16)
        plt.xlabel("평균 SHAP 값 (예측에 대한 기여도)", fontsize=12)
        plt.tight_layout()

        # 🔴 [수정] 파일명에 타임스탬프 추가
        output_filename = f"shap_summary_{analysis_type}_{run_timestamp}.png"
        output_path = os.path.join(paths["models"], output_filename)
        plt.savefig(output_path)
        plt.close()
        logger.info(f"SHAP Summary Plot이 '{output_path}'에 저장되었습니다.")
        
    # 5. 개별 분석 (Local Analysis)
    elif analysis_type == 'local':
        latest_pred_file = _get_latest_prediction_file(paths["predictions"])
        if not latest_pred_file:
            logger.error("분석할 추천 종목 파일이 없습니다. '1. 오늘의 추천 종목 생성'을 먼저 실행해주세요.")
            return
        
        logger.info(f"최신 추천 종목 파일 로드: {latest_pred_file}")
        recs_df = pd.read_csv(latest_pred_file, dtype={'code': str})
        pred_date_str = os.path.basename(latest_pred_file).replace('.csv', '')
        pred_date = pd.to_datetime(pred_date_str)

        features_today = load_partition_day(paths["features"], pred_date, pred_date)
        if features_today.empty:
            logger.error(f"{pred_date.date()} 날짜의 피처 데이터가 없습니다.")
            return

        recs_with_features = pd.merge(recs_df, features_today, on='code', how='left')
        
        report_dir = os.path.join(paths["backtest"], "shap_reports", pred_date_str)
        ensure_dir(report_dir)
        logger.info(f"개별 분석 리포트는 다음 폴더에 저장됩니다: {report_dir}")

        for _, row in recs_with_features.iterrows():
            X_rec = row[model.feature_name_].to_frame().T
            X_rec = X_rec.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            shap_values_local = explainer.shap_values(X_rec)
            
            if isinstance(explainer.expected_value, (list, np.ndarray)):
                expected_value = explainer.expected_value[1]
                shap_values_for_class1 = shap_values_local[1]
            else:
                expected_value = explainer.expected_value
                shap_values_for_class1 = shap_values_local

            plot_title = f"종목 {row['code']} ({row['name']}) SHAP 분석 (예측 점수: {row['score']:.2f})"
            _set_korean_font()
            
            shap.force_plot(expected_value, shap_values_for_class1, X_rec, matplotlib=True, show=False, text_rotation=15)
            plt.title(plot_title, fontsize=12)
            plt.tight_layout()
            
            filename = f"{int(row['rank']):02d}_{row['code']}_{row['name']}_shap.png"
            output_path = os.path.join(report_dir, filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        logger.info(f"총 {len(recs_with_features)}개 종목에 대한 SHAP Force Plot 생성을 완료했습니다.")

    logger.info("="*60)

