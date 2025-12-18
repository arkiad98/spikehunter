# addons/addon_cs_weight_finder.py

import pandas as pd
import numpy as np
from tabulate import tabulate
import copy

from modules.utils_io import read_yaml
from modules.utils_logger import logger
from modules.train import _run_classification_training

def run_find_optimal_cs_weight(settings_path: str):
    """
    여러 scale_pos_weight 값에 대한 분류 모델의 성능을 일괄 테스트하고 비교합니다.
    """
    logger.info("\n" + "="*80)
    logger.info("      <<< 최적 비용 민감도(scale_pos_weight) 탐색기 시작 >>>")
    logger.info("="*80)
    
    base_cfg = read_yaml(settings_path)
    
    # 테스트할 scale_pos_weight 값 목록
    weights_to_test = [2, 3, 4, 5, 15]
    
    all_results = []

    for weight in weights_to_test:
        logger.info("\n" + "="*80)
        logger.info(f"      - 테스트 시작: scale_pos_weight = {weight}")
        logger.info("="*80)
        
        # 현재 테스트에 사용할 설정 복사 및 수정
        temp_cfg = copy.deepcopy(base_cfg)
        if 'ml_params' not in temp_cfg:
            temp_cfg['ml_params'] = {}
        if 'lgbm_params_classification' not in temp_cfg['ml_params']:
            temp_cfg['ml_params']['lgbm_params_classification'] = {}
        temp_cfg['ml_params']['lgbm_params_classification']['scale_pos_weight'] = weight

        try:
            # _run_classification_training 함수를 재활용하되, 결과만 받아옵니다.
            # 함수 내부에서 로그가 출력되므로 여기서는 추가 출력을 최소화합니다.
            cv_results, _ = _run_classification_training(temp_cfg, return_results_only=True)
            
            if cv_results:
                result_row = {
                    'scale_pos_weight': weight,
                    'Avg Precision': np.mean(cv_results.get('average_precision', [0])),
                    'F1-Score': np.mean(cv_results.get('f1', [0])),
                    'Precision': np.mean(cv_results.get('precision', [0])),
                    'Recall': np.mean(cv_results.get('recall', [0])),
                    'Specificity': np.mean(cv_results.get('specificity', [0]))
                }
                all_results.append(result_row)
        except Exception as e:
            logger.error(f"scale_pos_weight = {weight} 테스트 중 오류 발생: {e}", exc_info=True)

    if not all_results:
        logger.error("결과를 수집하지 못했습니다. 테스트 과정에 문제가 없는지 확인해주세요.")
        return

    # 최종 결과 테이블로 출력
    results_df = pd.DataFrame(all_results).sort_values(by="Avg Precision", ascending=False)
    
    logger.info("\n" + "="*100)
    logger.info("                              <<< scale_pos_weight 값별 성능 비교 최종 결과 >>>")
    logger.info("="*100)
    print(tabulate(results_df, headers='keys', tablefmt='psql', floatfmt=".4f", showindex=False))
    logger.info("="*100)
    logger.info("* Avg Precision 또는 F1-Score가 가장 높은 지점이 일반적으로 가장 균형 잡힌 값입니다.")
    logger.info("* 이 결과를 바탕으로 settings.yaml의 scale_pos_weight 값을 수정하고, SMOTEENN(weight=1)과 최종 비교해주세요.")