
import sys

from typing import List

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool, tool
from langchain_tavily import TavilySearch

"""
    调用工具的流程: 
        定义工具 ——> 绑定工具 ——> 调用工具

    定义工具的2种方式:
        1. 使用 @tool 装饰器
        2. 使用 StructuredTool
"""


"""
    定义工具方式一 : 使用 @tool 装饰器
        使用要求：
            1. 函数定义 ——》 LLM 知道如何有这个函数
            2. 函数类型说明 ——》LLM 知道如何调用
            3. 文档字符串 ——》大模型知道有什么用
"""

@tool
def add(a : int , b : int ) -> int:
    """两数相加

    Args:
        a : 第一个整数参数
        b : 第二个整数参数
    """
    return a + b


# 打印 add 的相关信息
# print(add.invoke({"a": 1, "b": 2}))
# print(add.name)
# print(add.description)
# print(add.args)


"""
    补充：对于方式一 可以将文档字符串与函数定义分开存放
    有两种方式 : 
        1. 定义类
        2. 将说明放在参数位置
        
    Field 为字段声明额外的内容，其中 ... 表示该字段必填，无默认值（可使用 default= 来设置默认值，也可以设置范围约束，字符串格式约束），description 对字段的描述信息
"""
class AddInputBase(BaseModel):
    """两数相加"""
    a : int = Field(... , description="第一个整型参数")
    b : int = Field(... , description="第二个整型参数")

@tool(args_schema=AddInputBase)
def add2(a: int, b: int) -> int:
    return a + b

@tool
def add3(
    a: int = Field(..., description="第一个整型参数"),
    b: int = Field(..., description="第二个整型参数")
)->int:
    """两数相加"""
    return a + b


"""
    第二种定义工具的方式 : StructuredTool
"""
def add4(a: int, b: int) -> int:
    """两数相加"""
    return a + b
add_tool1 = StructuredTool.from_function(func = add4)
# print(add_tool1.invoke({"a": 2, "b": 3}))

"""
    对于第二种方式也可以 将字符串说明与函数定义分开
"""
def add5(a: int, b: int) -> int:
    return a + b
add_tool2 = StructuredTool.from_function(
    func = add5,
    name = "ADD" , # 可以对工具进行重命名
    description = "两数相加" ,    # 工具描述
    args_schema = AddInputBase , # 工具参数
)


"""
    调用工具的时候我们可以保留工具调用的过程
    加入 response_format 配置
"""
def add6(a: int, b: int) -> tuple[str , List[int]]:
        nums = [a , b]                   # 记录调用过程 , 参数是什么
        content = f"{nums}相加结果是{a + b}"   # 返回结果
        return content, nums

add_tool3 = StructuredTool.from_function(
    func = add6,
    description = "两数相加" ,    # 工具描述
    args_schema = AddInputBase , # 工具参数
    response_format = "content_and_artifact"    # 指明返回结果的格式
)
# print(add_tool3.invoke({"a": 2, "b": 3}))



"""
    - 绑定工具，让 LLM 可以是用我们设置的工具
        通过 .bind_tools() 进行绑定
"""
model = ChatOpenAI(model="gpt-4o-mini")
tools = [add]
model_with_tools = model.bind_tools(tools)

# 模型判断需要调用工具
result = model_with_tools.invoke("9加6等于多少？")
# print(result.tool_calls)
# [{'name': 'add', 'args': {'a': 9, 'b': 6}, 'id': 'call_xxx', 'type': 'tool_call'}]

# 模型判断不需要调用工具
result = model_with_tools.invoke("hello world!")
# print(result.content)


"""
    调用工具详细流程：
        将工具和问题交给模型 ——》需要调用工具 ——》模型返回工具调用的相关信息：使用到的工具以及结果 ——》将工具调用信息交给模型，获得最终答案
"""

@tool
def add_tool(a: int = Field(..., description="First integer"), b: int =  Field(..., description="Second integer")) -> int:
    """Add two integers."""
    return a + b

@tool
def multiply_tool(a: int = Field(..., description="First integer"), b: int =  Field(..., description="Second integer")) -> int:
    """Multiply two integers."""
    return a * b

all_tools = [add_tool , multiply_tool]

gptmodel = ChatOpenAI(model="gpt-4o-mini")
model_with_tools = gptmodel.bind_tools(all_tools)
messages = [
    HumanMessage("1加100等于多少 , 10乘以102等于多少") ,
]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

select_tool = {"add_tool" : add_tool , "multiply_tool" : multiply_tool}
for tool_call in ai_msg.tool_calls:
    tool_msg = select_tool[tool_call["name"]].invoke(tool_call)
    messages.append(tool_msg)

# result = gptmodel.invoke(messages)
# print(result.content)


"""
    - 调用 LangChain 提供的联网搜索tool
"""
web_tool = TavilySearch(max_results = 3)
model_wth_tavily = gptmodel.bind_tools(tools = [web_tool , ] , tool_choice="any")

messages = [
    HumanMessage("今天是几月几号呀")
]

ai_msg = model_wth_tavily.invoke(messages)
messages.append(ai_msg)

select_tool = {"add_tool" : add_tool , "multiply_tool" : multiply_tool , "tavily_search" : web_tool}

for tool_call in ai_msg.tool_calls:
    tool_msg = select_tool[tool_call["name"]].invoke(tool_call)
    messages.append(tool_msg)
print(gptmodel.invoke(messages).content)

