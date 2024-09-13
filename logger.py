import logging

# 创建一个日志器
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# 创建文件处理器
file_handler = logging.FileHandler('output.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)

# 创建格式器并添加到处理器
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 将处理器添加到日志器
logger.addHandler(console_handler)
logger.addHandler(file_handler)