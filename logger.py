# logger.py

import logging

def setup_logger(output_path):
    """
    配置日志器，将日志输出到指定文件和控制台。
    """
    # 创建日志器
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    # 创建处理器：文件处理器和控制台处理器
    file_handler = logging.FileHandler(output_path, mode='a', encoding='utf-8')
    console_handler = logging.StreamHandler()

    # 设置日志级别
    file_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.DEBUG)

    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 为处理器设置格式器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到日志器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 返回配置好的日志器
    return logger
