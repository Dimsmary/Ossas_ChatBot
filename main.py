import threading
import socket
import json
import time
import Seq2Seq as rnn
import requests
import os

states = [1, 1]
reply_dict = [None, None]
input_str = ['', '']

work_dir = rnn.works_dir
coolq_config_path = os.path.join(work_dir, 'coolq_config.txt')

# 读取配置文件
with open(coolq_config_path, 'r', encoding='UTF-8') as f:
    coolq_config = json.load(f)

url = coolq_config['post_url']
at_id = coolq_config['at_id']
host_port = coolq_config['host_port']


def http_receive():
    print(time.asctime() + ' 开始监听Coolq/HTTP上报')
    while True:
        cl = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 基于IPV4 TCP协议 TCP断开后立即释放
        cl.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 绑定地址 端口
        cl.bind(('', host_port))
        cl.listen(5)
        conn, address = cl.accept()
        data = conn.recv(10240).decode()

        # 搜索上报信息
        xn = data.find('\r\n\r\n')
        post = data[xn + 4:]
        try:
            post_dict = json.loads(post)
        except json.decoder.JSONDecodeError:
            post_dict = {}

        if post_dict != {}:
            print(post_dict)
            try:
                find = post_dict['message'].find(at_id)
            except KeyError:
                find = -1
            if find >= 0:
                str_in = post_dict['message'][22:]
                print(str_in)
                # 任务分配给线程
                while True:
                    if states[0]:
                        input_str[0] = str_in
                        reply_dict[0] = post_dict
                        states[0] = 0
                        break
                    # elif states[1]:
                    #     input_str[1] = str_in
                    #     reply_dict[1] = post_dict
                    #     states[1] = 0
                    #     break
                    else:
                        time.sleep(2)


def input_process(id):

    while True:
        if states[id]:
            time.sleep(2)
        else:
            in_dict = reply_dict[id]
            in_seq = input_str[id]
            print(time.asctime() + ' 线程' + str(id) + '正在处理消息')
            output = rnn.predict(in_seq)
            data = {"group_id": in_dict['group_id'],
                    "message": "[CQ:at,qq=" + str(in_dict['sender']['user_id']) + "] " + output}
            requests.post(url, data=data)
            states[id] = 1


print("------------------------")
print("---OSSAS ChatBot-----")
print("--人工智障聊天机器人---")
print("---版本：0.0.3_alpha---")
print("-------------------------")
print("作者：Dimsmary")
print("网址：dimsmary.tech")
print("-----------------")
print("本软件完全免费")
print("代码遵循MIT协议")
print("-----------------")
print("")
mode = '0'
while True:
    print("-----------------")
    print("模式1：搭建模型")
    print("模式2：训练模型")
    print('模式3：开启Coolq接口')
    print("模式4：进行对话")
    print("-----------------")
    mode = input('输入工作模式：')
    if mode == '1':
        rnn.pre_precess()
        rnn.setup_model()
    elif mode == '2':
        epo = input('输入循环轮数：')
        bat = input('输入batch size:')
        rnn.train_model(bat, epo)
    elif mode == '3':
        t0 = threading.Thread(target=input_process, args=(0,), name='http_receive0')
        t0.start()
        http_receive()
    elif mode == '4':
        print('输入数字0 退出')
        while True:
            str_in = input('你说：')
            if str_in == '0':
                break
            print(rnn.predict(str_in))

    else:
        print('输入有误')
