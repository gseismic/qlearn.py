import sys
from loguru import logger

def make_logger(name=None, file=None, 
                format="{time: YYYY-MM-DD HH:mm:ss} | {level} | {message}", 
                level="INFO"):
    log = logger.bind(name=name) if name else logger
    log.remove()
    log.add(sys.stdout, format=format, level=level)
    if file is not None:
        log.add(file, format=format, level=level)
    return log

sys_logger = make_logger("system", level="DEBUG")
user_logger = make_logger("user", level="INFO")
