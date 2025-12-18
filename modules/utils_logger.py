# modules/utils_logger.py (ver 2.7.00)
"""프로젝트 전반에서 사용될 고도화된 로깅 시스템 모듈.

"""
import os
import sys
import logging
import re
from datetime import datetime

try:
    from pythonjsonlogger import jsonlogger
except ImportError:
    print("Error: 'python-json-logger' is not installed. Please install it using 'pip install python-json-logger'")
    sys.exit(1)

import warnings
# pkg_resources deprecation warning suppression
warnings.filterwarnings("ignore", category=UserWarning, module='pykrx') 
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# 🔴 [추가] 비표준 로그 레코드를 사전에 처리하는 필터 클래스
class ExternalLibFilter(logging.Filter):
    """
    pykrx와 같이 비표준 인자로 로그를 생성하는 외부 라이브러리의
    로그 레코드를 포매터가 처리하기 전에 안전하게 수정합니다.
    """
    def filter(self, record):
        try:
            # getMessage()가 성공하는지 테스트. 실패 시 TypeError 발생.
            record.getMessage()
        except TypeError:
            # 실패 시, msg와 args를 안전한 문자열로 조합하여 msg에 덮어쓰고 args를 비움.
            record.msg = f"Unformattable message: msg={str(record.msg)}, args={str(record.args)}"
            record.args = ()
        return True

class CleanJsonFormatter(jsonlogger.JsonFormatter):
    """
    ANSI 코드를 제거하는 커스텀 JSON 포매터.
    """
    def format(self, record):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        
        # 이제 getMessage()는 필터에서 안전하게 처리되었으므로 바로 호출 가능
        message = record.getMessage()
        record.message = ansi_escape.sub('', message)

        if record.exc_info:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        else:
            record.exc_text = None
            
        return super().format(record)

class StreamToLogger:
    """
    표준 출력(stdout) 스트림을 로깅 시스템으로 재지정하는 클래스.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.original_stream = sys.__stdout__

    def write(self, buf):
        if '\r' in buf and ('it/s' in buf or '%' in buf):
            self.original_stream.write(buf)
            self.original_stream.flush()
        else:
            for line in buf.rstrip().splitlines():
                if line:
                    self.logger.log(self.level, line.rstrip())

    def flush(self):
        self.original_stream.flush()

def handle_exception(exc_type, exc_value, exc_traceback):
    """처리되지 않은 예외를 전역적으로 처리하고 로깅하는 핸들러."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    if logger:
        logger.critical(
            "Unhandled exception caught by global handler", 
            exc_info=(exc_type, exc_value, exc_traceback)
        )

def setup_global_logger(run_timestamp: str):
    """
    프로젝트 전역에서 사용될 로거를 설정하고 초기화합니다.
    """
    if logger.hasHandlers():
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    console_handler = logging.StreamHandler(sys.__stdout__)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    log_filename = f"{run_timestamp}_pipeline.json"
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename), encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    json_formatter = CleanJsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(filename)s %(lineno)d %(message)s',
        json_ensure_ascii=False
    )
    file_handler.setFormatter(json_formatter)
    
    # 🔴 [추가] 파일 핸들러에 외부 라이브러리용 필터 추가
    file_handler.addFilter(ExternalLibFilter())

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    root_logger = logging.getLogger()
    if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        root_logger.setLevel(logging.INFO)
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
        root_logger.addHandler(file_handler)
        # 🔴 [추가] 루트 로거에도 필터를 추가하여 모든 외부 라이브러리 로그에 적용
        root_logger.addFilter(ExternalLibFilter())
    
    sys.excepthook = handle_exception
    
    logger.info(f"Logger initialized. All logs for this run will be saved to: logs/{log_filename}")
    
    sys.stdout = StreamToLogger(logger, logging.INFO)
    
    return logger

logger = logging.getLogger("QuantPipeline")
logger.setLevel(logging.DEBUG)
logger.propagate = False

