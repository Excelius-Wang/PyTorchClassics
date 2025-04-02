import datetime
import os
import time

# 创建日志目录
os.makedirs('logs', exist_ok=True)

# 创建一个唯一的日志文件名
timestamp = time.strftime('%Y%m%d_%H%M%S')
log_file = os.path.join('logs', f'training_log_{timestamp}.txt')


def save_log(message):
    """将日志信息保存到文件
    
    Args:
        message: 要记录的信息
    """
    with open(log_file, 'a', encoding='utf-8') as f:
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"==========={nowtime}===========\n")
        f.write(f"{str(message)}\n\n")


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("==========" * 8 + "%s" % nowtime)
    print(str(info))

    # 同时保存到日志文件
    save_log(info)
