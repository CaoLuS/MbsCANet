"""
    @Author: Panke
    @Time: 2023-04-28  17:29
    @Email: None
    @File: test.py
    @Project: MbsCANet
"""
import openai

openai.api_key = "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"


# 通过 `系统(system)` 角色给 `助手(assistant)` 角色赋予一个人设
messages = [{'role': 'system', 'content': '你是一个乐于助人的诗人。'}]
# 在 messages 中加入 `用户(user)` 角色提出第 1 个问题
messages.append({'role': 'user', 'content': '作一首诗，要有风、要有肉，要有火锅、要有雾，要有美女、要有驴！'})
# 调用 API 接口
response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=messages,
)
# 在 messages 中加入 `助手(assistant)` 的回答
messages.append({
    'role': response['choices'][0]['message']['role'],
    'content': response['choices'][0]['message']['content'],
})
# 在 messages 中加入 `用户(user)` 角色提出第 2 个问题
messages.append({'role': 'user', 'content': '好诗！好诗！'})
# 调用 API 接口
response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=messages,
)
# 在 messages 中加入 `助手(assistant)` 的回答
messages.append({
    'role': response['choices'][0]['message']['role'],
    'content': response['choices'][0]['message']['content'],
})
# 查看整个对话
print(messages)