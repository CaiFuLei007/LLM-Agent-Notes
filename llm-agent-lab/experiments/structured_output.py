from typing import List, Union

from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from pydantic import Field
from pydantic.v1.schema import json_scheme
from tenacity import WrappedFn
from typing_extensions import TypedDict, Annotated

"""
    LangChain 聊天模型提供了结构化输出的能力
    传统的聊天模型返回的是字符串，对于程序代码编写并不友好

    聊天模型的 with_structured_output 方法让我们可以预先新定义一个期望的数据结构
    我们可以让模型给我们返回 : Pydantic 对象 ， Json 结构， TypedDict 字典
    下面一次对这几种方式进行举例
"""

"""
    返回 Pydantic 对象
"""
gptmodel = ChatOpenAI(model="gpt-4o-mini")


class WeatherInfo(BaseModel):
    city : str = Field(... , description="城市名称")
    temperature : str = Field(... , description="当前气温")
    condition : str = Field(... , description="当前情况")
    humidity : str = Field(... , description="适度百分比")

structured_weatherinfo_model = gptmodel.with_structured_output(WeatherInfo)

# result = structured_weatherinfo_model.invoke("查询北京今天的天气")
# print(result)

# 支持嵌套输出
class WeatherInfos(BaseModel):
    infos : List[WeatherInfo]

structured_weatherinfos_model = gptmodel.with_structured_output(WeatherInfos)
# result = structured_weatherinfos_model.invoke("查询北京和上海今天的天气")
# print(result)



"""
    返回 TypedDict 字典
"""

class WeatherInfo2(TypedDict):
    city : Annotated[str , ... , "城市名称"]
    temperature : Annotated[str , ... , "城市名称"]
    condition : Annotated[str , ... , "天气情况"]
    humidity : Annotated[str , ... , "湿度百分比"]

structured_weatherinfo_model2 = gptmodel.with_structured_output(WeatherInfo2)
# print(structured_weatherinfo_model2.invoke("查询北京今天的天气"))


"""
    返回 JSON
"""

json_scheme = {
    "title" : "weather_info" ,
    "description" : "查询天气情况" ,
    "type" : "object",
    "properties" : {
        "city" : {"type" : "string" , "description" : "城市名称"},
        "temperature" : {"type" : "string" , "description" : "当前湿度"},
        "condition" : {"type" : "string" , "description" : "天气状况"} ,
        "humidity" : {"type" : "string" , "description" : "湿度情况" , "default" : None}
    },
    "required": ["city", "temperature", "condition"],
}
# structured_weatherinfo_model3 = gptmodel.with_structured_output(json_scheme)
# print(structured_weatherinfo_model3.invoke("查询北京今天的天气"))


"""
    我们可能给模型发送各种各样的消息，模型如果只按照你的模板进行回复，结果可能出现错误
    因此我们需要让模型能够自己去根据问题，选择我们提供的结构化输出
"""

class ConversationalResponse(BaseModel):
    """以对话的方式回应。待人友善，乐于助人"""
    response: str = Field(description="对用户查询的会话响应")

class StoryResponse(BaseModel):
    """给用户讲一个简短的故事，字数50字以内"""
    begining : str = Field(description="故事的开头")
    climax : str = Field(description="故事高潮")
    num : int = Field(description="故事总字数")

# 对数据格式进行整合
class FinalResponse(BaseModel):
    final_output : Union[ConversationalResponse, StoryResponse , WeatherInfos]

structured_model = gptmodel.with_structured_output(FinalResponse)
print(structured_model.invoke("今天上海的天气怎么样"))
print(structured_model.invoke("讲一个关于小猫的故事"))
print(structured_model.invoke("你好"))