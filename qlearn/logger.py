from loguru import logger


# 创建一个名为 sys_logger 的 logger，并配置输出到文件 'system.log'
sys_logger = logger.bind(name="sys_logger")
#sys_logger.add(
#    "system.log",  # 输出到 'system.log' 文件
#    level="INFO",  # 日志级别为 INFO
#    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"  # 自定义日志格式
#)

# 创建一个名为 user_logger 的 logger，并配置输出到文件 'user.log'
user_logger = logger.bind(name="user_logger")
#user_logger.add(
#    "user.log",  # 输出到 'user.log' 文件
#    level="DEBUG",  # 日志级别为 DEBUG
#    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}"  # 不同的日志格式
#)
