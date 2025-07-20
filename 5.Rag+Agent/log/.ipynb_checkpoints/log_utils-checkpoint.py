import sys, os
from loguru import logger

# 获得当前项目的绝对路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_dir = os.path.join(root_dir, "logs")  # 存放项目日志目录的绝对路径

if not os.path.exists(log_dir):  # 如果日志目录不存在，则创建
    os.mkdir(log_dir)

# LOG_FILE = "translation.log"  # 存储日志的文件（若需启用文件日志，可取消注释并完善逻辑 ）

# Trace < Debug < Info < Success < Warning < Error < Critical
class MyLogger:
    def __init__(self):
        # log_file_path = os.path.join(log_dir, LOG_FILE)  # 若启用文件日志，取消该行注释并用于 logger.add 
        self.logger = logger  # 写日志的对象
        # 清空所有设置
        self.logger.remove()
        # 添加控制台输出的格式，sys.stdout 为输出到屏幕；关于这些配置还需要自定义请移步官网查看相关参数 
        self.logger.add(sys.stdout, level='DEBUG',
                        format="<green>{time:YYYYMMDD HH:mm:ss}</green> | "  # 颜色>时间
                               "{process.name} | "  # 进程名
                               "{thread.name} | "  # 线程名
                               "<cyan>{module}</cyan>.<cyan>{function}</cyan>"  # 模块名.方法名
                               ":<cyan>{line}</cyan> | "  # 行号
                               "<level>{level}</level>: "  # 等级
                               "<level>{message}</level>",  # 日志内容
                        )
        # 输出到文件的格式，注释下面的 add，则关闭日志写入（若需启用，取消注释并完善，如补充 log_file_path 等 ）
        # self.logger.add(log_file_path, level='DEBUG', encoding='UTF-8',
        #                 format='{time:YYYYMMDD HH:mm:ss} - '  # 时间
        #                        '{process.name} | '  # 进程名
        #                        '{thread.name} | '  # 线程名 
        #                        '{module}.{function}:{line} - {level} - {message}',  # 日志格式
        #                 rotation="10 MB",  # 日志文件生成的规则，如按大小、按时间等 
        #                 retention=20  # 保留日志文件的规则
        #                 )

    def get_logger(self):
        return self.logger

log = MyLogger().get_logger()

if __name__ == '__main__':
    # 以下为测试日志输出，可根据需要启用
    # log.debug("This is a debug message.")
    # log.info("This is an info message.")
    pass