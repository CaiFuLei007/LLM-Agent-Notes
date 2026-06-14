
import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 定义模型
model = ChatOpenAI(model="gpt-4o-mini")

# 定时消息
message = [
    SystemMessage("帮我将下面的句子翻译为中文") ,
    HumanMessage("hello")
]

# 定义解析器
parse = StrOutputParser()

# 定义链
chain = model | parse

# 输出结果
print(chain.invoke(message))