"""
    @Author: Panke
    @Time: 2022-11-09  19:17
    @Email: None
    @File: Service.py
    @Project: MbsCANet
"""

import json
import socket
import threading
import Diagnose

import Test

# 文件名
nnservice = Diagnose


class ServerThreading(threading.Thread):
    def __init__(self, clientsocket, recvsize=1024 * 1024, encoding="utf-8"):
        threading.Thread.__init__(self)
        self._socket = clientsocket
        self._recvsize = recvsize
        self._encoding = encoding
        pass

    def run(self):
        print("开启线程")
        try:
            # 接受数据
            msg = ''
            while True:
                # 读取recvsize个字节
                rec = self._socket.recv(self._recvsize)
                # 解码
                msg += rec.decode(self._encoding)
                # 文本接受是否完毕，因为python socket不能自己判断文本接受数据是否完毕，所以需要自定义协议标志数据接受完毕
                if msg.strip().endswith('over'):
                    msg = msg[:-4]
                    break
            print("接收数据完成：", json.loads(msg))
            # 解析json格式的数据
            index = 'X'
            re = json.loads(msg)
            img = re['img']
            magnification = re['magnification']
            magnification = magnification[:magnification.index(index)]
            # 调用方法
            print("诊断中...")
            # heat_map = heatmap(img)
            res = nnservice.diagnose(img, magnification)
            sendmsg = json.dumps(res)
            # 发送数据
            self._socket.send(("%s" % sendmsg).encode(self._encoding))
            pass
        except Exception as identifier:
            self._socket.send("500".encode(self._encoding))
            print(identifier)
            pass
        finally:
            self._socket.close()
        print("任务结束")
        return res
        pass

    def __del__(self):
        pass


def main():
    # 创建服务器套接字
    serverScoket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 获取本地主机名
    host = socket.gethostname()
    # 设置端口号
    port = 12345
    # 将套接字与本机主机和端口绑定
    serverScoket.bind((host, port))
    # 监听最大数量
    serverScoket.listen(5)
    # 获取本机的连接信息
    myaddr = serverScoket.getsockname()
    print("服务器地址%s" % str(myaddr))
    # 循环等待接受客户端的消息
    while True:
        # 获取一个客户端的连接
        clientsocket, addr = serverScoket.accept()
        print("连接地址：", str(addr))
        try:
            t = ServerThreading(clientsocket)  # 为每个请求开启一个处理线程
            t.start()
            pass
        except Exception as identifier:
            print(identifier)
            break
            pass
        pass
    serverScoket.close()
    pass


if __name__ == '__main__':
    main()
