"""
settings.yaml 파일의 구조와 각 파라미터의 데이터 타입을 정의하는 스키마.
이 스키마는 설정 파일의 유효성을 검사하는 데 사용됩니다.
"""

# 각 파라미터에 대한 규칙을 정의합니다.
# 'type': 기대하는 데이터 타입 (int, float, str, list, dict)
# 'required': True이면 해당 파라미터가 반드시 존재해야 함
# 'schema': 중첩된 딕셔너리의 경우, 하위 스키마를 정의

SETTINGS_SCHEMA = {
    'top_n': {'type': int, 'required': True},
    'min_avg_value': {'type': int, 'required': True},
    'fee_rate': {'type': float, 'required': True},
    'backtest_defaults': {
        'type': dict, 'required': True, 'schema': {
            'initial_cash': {'type': float, 'required': True},
            'default_years': {'type': int, 'required': True},
            'min_ml_target_return': {'type': float, 'required': True}
        }
    },
    'walk_forward': {
        'type': dict, 'required': True, 'schema': {
            'total_start_date': {'type': str, 'required': True},
            'train_months': {'type': int, 'required': True},
            'test_months': {'type': int, 'required': True}
        }
    },
    'paths': {
        'type': dict, 'required': True, 'schema': {
            'raw_prices': {'type': str, 'required': True},
            'raw_fundflow': {'type': str, 'required': True},
            'merged': {'type': str, 'required': True},
            'features': {'type': str, 'required': True},
            'backtest': {'type': str, 'required': True},
            'predictions': {'type': str, 'required': True},
            'ml_dataset': {'type': str, 'required': True},
            'models': {'type': str, 'required': True},
            'meta': {'type': str, 'required': True},
            'cache': {'type': str, 'required': True}
        }
    },
    'strategies': {
        'type': dict, 'required': True, 'schema': {
            # 모든 SpikeHunter_R* 전략에 대해 동일한 구조를 검사
            '*': {
                'type': dict, 'required': True, 'schema': {
                    'name': {'type': str, 'required': True},
                    'target_r': {'type': float, 'required': True},
                    'stop_r': {'type': float, 'required': True},
                    'max_hold': {'type': int, 'required': True},
                    'max_dist_from_ma': {'type': float, 'required': True},
                    'entry_slip_r': {'type': float, 'required': True},
                    'trail_k': {'type': float, 'required': True},
                    'max_market_vol': {'type': float, 'required': True},
                    'max_daily_ret_entry': {'type': float, 'required': True},
                    'min_ml_score': {'type': float, 'required': True},
                    'min_avg_value': {'type': float, 'required': True}, # float 허용 (e.g. 1e9)
                    'top_n': {'type': int, 'required': True}
                }
            }
        }
    },
    'ml_params': {
        'type': dict, 'required': True, 'schema': {
            'target_surge_rate': {'type': float, 'required': True},
            'target_hold_period': {'type': int, 'required': True},
            'regression_sampling_quantile': {'type': float, 'required': True},
            'lgbm_params_classification': {'type': dict, 'required': True},
            'lgbm_params_regression': {'type': dict, 'required': True}
        }
    },
    'optimization': {
        'type': dict, 'required': True, 'schema': {
            'SpikeHunter': {
                'type': dict, 'required': True, 'schema': {
                    'optimize_on': {'type': str, 'required': True},
                    'n_trials': {'type': int, 'required': True},
                    'param_space': {'type': dict, 'required': True}
                }
            }
        }
    },
    'ml_optimization': {
        'type': dict, 'required': True, 'schema': {
            'classification': {
                'type': dict, 'required': True, 'schema': {
                    'optimize_on': {'type': str, 'required': True},
                    'n_trials': {'type': int, 'required': True},
                    'param_space': {'type': dict, 'required': True}
                }
            },
            'regression': {
                'type': dict, 'required': True, 'schema': {
                    'optimize_on': {'type': str, 'required': True},
                    'n_trials': {'type': int, 'required': True},
                    'param_space': {'type': dict, 'required': True}
                }
            }
        }
    }
}
