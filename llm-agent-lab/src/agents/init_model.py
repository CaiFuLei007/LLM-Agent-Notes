
import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model

"""
创建模型的 2 种方式:
    1. 直接使用 ChatOpenAI 进行创建
    2. 使用 init_chat_model 进行创建
"""


"""
    方式一 : 使用 ChatOpenAI 
    
    类型 : langchain_openai_chat_models.base.ChatOpenAI
            | 继承于
          langchain_openai_chat_models.base.BaseChatOpenAI
            | 继承于
          Runnable : 统一 invoke 操作
"""
openai_model = ChatOpenAI(model="gpt-4o-mini")


"""
    方式二 : 使用 init_chat_model 
"""

deepseek_model = init_chat_model(model = "deepseek-v4-flash",model_provider = "deepseek")

"""
    补充 : 模型配置器 
        通过 init_chat_model 定制可配置模型
        
    1. 设置好统一的配置信息
    2. 发送消息的时候再配置模型
"""
config_model = init_chat_model(temperature = 0.2)

message = [
    SystemMessage("帮我将下面的句子翻译为中文") ,
    HumanMessage("hello")
]

result = config_model.invoke(input = message , config = {
    "configurable" :{
        "model":"deepseek-v4-flash",
    }
})
print(result)


""" 
    对于可配置模型，如果想要在后续修改默认参数，使用 configable_fields ，比如修改上面的 temperature = 0.2
"""
config_model2 = init_chat_model(
    temperature = 0.2 ,
    configurable_fields=("temperature" , "model"), # "any" 表示所有都可以进行修改
    config_prefix="first"
)

result = config_model2.invoke(
    input = message ,
    config = {
        "configurable" :{
            "first_temperature":2 ,
            "first_model":"deepseek-v4-flash",
        }
    }
)
print(result)





