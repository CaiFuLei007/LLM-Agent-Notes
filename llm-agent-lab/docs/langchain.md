

# LangChain 

## LLM 存在的六大问题

### 幻觉问题
- LLM 不知道答案，瞎编答案

解决方案：添加外部知识库，将知识库的内容和问题都交给 LLM ，让 LLM 从外部知识库中寻找答案

### 无实时性
- 大模型训练有截止日期，无实时数据

解决方案：外部添加数据库，使用 LangChain 工具联网搜索

### 切换模型麻烦
- 不同模型的 API 接口不一样，请求和响应字段不一样，要对每一个模型都单独进行字段解析

解决方案：LangChain 已经将接口进行统一

### 无结构化数据
- 从模型获取到的消息是一段话，而不是结构化数据，不便于编程

比如有一个 患病诊断的 LLM ，我们希望从 LLM 那里获取到的响应的形式：
```json
{
  "condition" : [
    "普通感冒" ,
    "流感"
  ],
  "advice" : [
    "多喝水" , 
    "打点滴"
  ]
}
```
如果直接将消息发送给大模型，难以实现

解决方法：使用 LangChain 的 Output Parsers（输出解析器）将模型输出解析为结构化数据

### 统一的提示词
- 提示词的质量和风格，决定了输出结果的准确性
- 给大模型添加系统提示词，比如有一个 患病诊断的AI 应用，用户使用的时候会直接告诉 LLM 其身体状况。因此我们需要添加系统提示词，来补充大模型的上下文信息

解决方法：使用 LangChain 封装的 Message 字段

### LLM 从指定位置查找
- 我们希望 LLM 从企业内部数据中获取答案

解决方法：给大模型添加数据库，使用 LangChain 工具


## LangChain 介绍

> 开发 AI 应用程序的框架
> 将自然语言处理的流程拆解为标准化的组件，让开发者能够自由组合，定制

核心设计：以**链式**的方式，整合多个组件

## 模型创建

LangChain 提供了 **2 种方式** 创建模型：

### 方式一：使用 ChatOpenAI 直接创建

适用于只需要调用 OpenAI 系列模型的场景，直接实例化即可：

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
```

**继承关系：**

```
ChatOpenAI
  └── BaseChatOpenAI
        └── Runnable  →  统一 invoke 操作
```

`ChatOpenAI` 继承自 `Runnable`，因此可以直接使用 `.invoke()` 方法调用模型。

### 方式二：使用 init_chat_model 统一创建

LangChain 提供了 `init_chat_model` 工厂函数，可以用统一的接口创建**不同供应商**的模型，避免记忆各个供应商的类名：

```python
from langchain.chat_models import init_chat_model

# 创建 DeepSeek 模型
model = init_chat_model(model="deepseek-v4-flash", model_provider="deepseek")

# 创建 OpenAI 模型
model = init_chat_model(model="gpt-4o-mini", model_provider="openai")
```

只需切换 `model` 和 `model_provider` 参数，就能在不同模型之间无缝切换。

### 补充：模型配置器（Configurable Model）

通过 `init_chat_model` 可以创建**可配置模型**，在定义时设置默认参数，在调用时再动态指定模型和参数，适合需要灵活切换模型的场景。

**基本用法：** 创建时设置默认配置，调用时通过 `config` 覆盖：

```python
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model

# 创建时设置默认参数
config_model = init_chat_model(temperature=0.2)

message = [
    SystemMessage("帮我将下面的句子翻译为中文"),
    HumanMessage("hello")
]

# 调用时再指定模型
result = config_model.invoke(input=message, config={
    "configurable": {
        "model": "deepseek-v4-flash"
    }
})
```

**进阶用法：** 使用 `configurable_fields` 将指定参数暴露为可配置项，配合 `config_prefix` 避免参数名冲突：

```python
config_model2 = init_chat_model(
    temperature=0.2,
    configurable_fields=("temperature", "model"),  # "any" 表示所有参数都可配置
    config_prefix="first"  # 配置时需加前缀，避免命名冲突
)

# 调用时通过 "前缀_参数名" 的方式覆盖
result = config_model2.invoke(
    input=message,
    config={
        "configurable": {
            "first_temperature": 2,
            "first_model": "deepseek-v4-flash"
        }
    }
)
```

## 调用工具

在 LangChain 中，聊天模型提供了**工具调用**功能，能使 LLM 与外部服务、API 和数据库进行交互，也可用于从非结构化数据中提取结构化信息。

例如 LLM 无法获取实时天气，此时可以借助工具通过外部搜索服务完成查询；LLM 无法直接获取数据库表数据，也可以借助工具与数据库交互完成查询。

### 创建工具

#### 使用 @tool 装饰器

`@tool` 装饰器是自定义工具的最简单方法，工具通过 `@tool` 加 Python 函数实现：

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers.

    Args:
        a: First integer
        b: Second integer
    """
    return a * b

print(multiply.invoke({"a": 2, "b": 3}))  # 输出：6
print(multiply.name)                      # 输出：multiply
print(multiply.description)               # 输出：Multiply two integers.
print(multiply.args)                      # 输出：{'a': {'title': 'A', 'type': 'integer'}, ...}
```

- 装饰器默认使用**函数名称**作为工具名称
- 装饰器使用**文档字符串**作为工具描述
- **函数名、类型提示和文档字符串**都是工具 Schema 的一部分，不可缺失

> **什么是 Schema？**
> Schema 是描述其他数据结构的声明格式，用于自动验证数据。工具 Schema 从函数名、类型提示和文档字符串中获取相关属性，以此声明工具的名称、描述、输入参数、输出类型等。
>
> 如果使用简单方式定义工具（如上述示例），工具 Schema 需要解析 **Google 风格的文档字符串**来获取参数描述：
>
> ```python
> def fetch_data(url, retries=3):
>     """从给定的URL获取数据。
>
>     Args:
>         url (str): 要从中获取数据的URL。
>         retries (int, optional): 失败时重试的次数。默认为3。
>
>     Returns:
>         dict: 从URL解析的JSON响应。
>     """
> ```

**其他定义模式：**

当使用 `@tool` 定义工具时没有提供文档字符串，会报错 `ValueError: Function must have a docstring`。此时有两种解决方式：

**模式一：依赖 Pydantic 类** — 使用 `args_schema` 参数，通过 `Field(description="...")` 添加字段描述：

```python
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class AddInput(BaseModel):
    """Add two integers."""
    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")

@tool(args_schema=AddInput)
def add(a: int, b: int) -> int:
    # 未提供文档字符串，依赖 Pydantic 类
    return a + b
```

**模式二：依赖 Annotated** — 使用 `Annotated` 和文档字符串传递给工具 Schema：

```python
from langchain_core.tools import tool
from typing_extensions import Annotated

@tool
def add(
    a: Annotated[int, ..., "First integer"],
    b: Annotated[int, ..., "Second integer"]
) -> int:
    """Add two integers."""
    return a + b
```

#### 使用 StructuredTool 类

`StructuredTool.from_function()` 类方法通过给定的函数来创建并返回一个工具：

```python
from langchain_core.tools import StructuredTool

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

calculator_tool = StructuredTool.from_function(func=multiply)
print(calculator_tool.invoke({"a": 2, "b": 3}))  # 输出：6
```

**关键参数：**

| 参数 | 说明 |
|------|------|
| `func` | 工具函数 |
| `name` | 工具名称，默认为函数名称 |
| `description` | 工具描述，默认为函数文档字符串 |
| `args_schema` | 工具输入参数的 Schema |
| `response_format` | 响应格式，默认 `"content"` |

**response_format 说明：**

- `"content"`（默认）：工具输出仅为 `ToolMessage` 的 `content` 属性（文本），供模型理解和推理
- `"content_and_artifact"`：输出为 `(content, artifact)` 元组，`artifact` 保存原始结构化数据，供链中后续组件使用

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import List, Tuple

class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")

def multiply(a: int, b: int) -> Tuple[str, List[int]]:
    nums = [a, b]
    content = f"{nums}相乘的结果是{a * b}"
    return content, nums

calculator_tool = StructuredTool.from_function(
    func=multiply,
    name="Calculator",
    description="两数相乘",
    args_schema=CalculatorInput,
    response_format="content_and_artifact"
)

# 直接调用，只返回 content
print(calculator_tool.invoke({"a": 2, "b": 3}))  # 输出：[2, 3]相乘的结果是6

# 模拟模型调用，返回 ToolMessage（包含 content 和 artifact）
print(calculator_tool.invoke({
    "name": "Calculator",
    "args": {"a": 2, "b": 3},
    "id": "123",
    "type": "tool_call",
}))
# 输出：ToolMessage(content='[2, 3]相乘的结果是6', name='Calculator', tool_call_id='123', artifact=[2, 3])
```

### 绑定工具

使用聊天模型的 `.bind_tools()` 方法将工具绑定到模型，返回一个 `Runnable` 实例：

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")
tools = [add, multiply]
model_with_tools = model.bind_tools(tools)
```

**`bind_tools()` 关键参数：**

| 参数 | 说明 |
|------|------|
| `tools` | 工具定义列表，支持字典、Pydantic 类、Python 函数、BaseTool |
| `tool_choice` | 控制模型调用哪个工具 |
| `strict` | `True` 保证模型输出与 JSON Schema 完全匹配 |
| `parallel_tool_calls` | `False` 禁用并行工具调用 |

**`tool_choice` 可选值：**
- `"auto"` — 自动选择工具（包括无工具）
- `"none"` — 不调用工具
- `"any"` / `"required"` / `True` — 强制调用至少一个工具
- `"<<tool_name>>"` — 调用指定工具

### 工具调用

绑定工具后，使用 `.invoke()` 完成工具调用。模型会根据输入的相关性**自主决定**是否调用工具：

```python
# 模型判断需要调用工具
result = model_with_tools.invoke("9乘6等于多少？")
print(result.tool_calls)
# [{'name': 'multiply', 'args': {'a': 9, 'b': 6}, 'id': 'call_xxx', 'type': 'tool_call'}]

# 模型判断不需要调用工具
result = model_with_tools.invoke("hello world!")
print(result.content)  # 输出：Hello! How can I assist you today?
```

输出结果是一个 `AIMessage`，其 `tool_calls` 属性包含工具调用所需的一切信息（工具名称、输入参数）。

### 强制模型调用工具

设置 `tool_choice="any"` 可强制模型调用至少一个工具：

```python
model_with_tools = model.bind_tools(tools, tool_choice="any")
result = model_with_tools.invoke("hello world!")
# 即使输入与工具无关，模型仍会调用工具
```

### 将工具输出传递给聊天模型

仅调用工具后，模型并未返回最终答案。需要将工具输出构建为 `ToolMessage`，连同之前的 `HumanMessage`、`AIMessage` 一起发送给模型，才能获得最终回答。

完整流程：

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from typing_extensions import Annotated

model = ChatOpenAI(model="gpt-4o-mini")

@tool
def add(a: Annotated[int, ..., "First integer"], b: Annotated[int, ..., "Second integer"]) -> int:
    """Add two integers."""
    return a + b

@tool
def multiply(a: Annotated[int, ..., "First integer"], b: Annotated[int, ..., "Second integer"]) -> int:
    """Multiply two integers."""
    return a * b

tools = [add, multiply]
model_with_tools = model.bind_tools(tools)

# 第一次调用：模型返回工具调用请求
messages = [HumanMessage("9乘6等于多少？5加3等于多少？")]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

# 执行工具调用，将结果构建为 ToolMessage
selected_tool = {"add": add, "multiply": multiply}
for tool_call in ai_msg.tool_calls:
    tool_msg = selected_tool[tool_call["name"]].invoke(tool_call)
    messages.append(tool_msg)

# 第二次调用：模型根据工具结果返回最终答案
result = model.invoke(messages)
print(result.content)  # 输出：9乘6等于54，5加3等于8。
```

**流程总结：**

```
第一次调用：HumanMessage → 模型 → AIMessage（含工具调用请求）
                                            ↓
                                    执行工具 → ToolMessage
                                            ↓
第二次调用：HumanMessage + AIMessage + ToolMessage → 模型 → AIMessage（最终答案）
```

### LangChain 提供的工具

LangChain 官方提供了大量现成的工具（Tool）和工具包（Toolkit），继承自 `BaseTool` 和 `BaseToolkit`，涵盖搜索、数据库、网页浏览器等功能，可直接使用。

#### TavilySearch 搜索工具

Tavily 是一个专门为 AI 设计的搜索引擎，能以结构化形式返回搜索结果，便于直接用于后续推理。

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch

model = ChatOpenAI(model="gpt-4o-mini")

# 绑定搜索工具，max_results 控制返回结果数量
tool = TavilySearch(max_results=4)
model_with_tools = model.bind_tools([tool])

messages = [HumanMessage("中国西安今天的天气怎么样？")]
ai_msg = model_with_tools.invoke(messages)
messages.append(ai_msg)

for tool_call in ai_msg.tool_calls:
    tool_msg = tool.invoke(tool_call)
    messages.append(tool_msg)

result = model_with_tools.invoke(messages)
print(result.content)
# 输出：今天西安的天气情况如下：晴天转多云，最高气温约31℃，最低气温约21℃...
```

## 结构化输出

在 LangChain 中，聊天模型提供了**结构化输出**功能，一种使聊天模型以结构化格式（例如 JSON）进行响应的技术。

### 核心概念

**从"字符串"到"对象"的范式转换：**

传统方式调用聊天模型得到的是一个 `AIMessage`，其内容是一个字符串：
```python
model = ChatOpenAI()
response = model.invoke("告诉我关于苹果公司的最新消息。")
print(response.content)
# 输出: "苹果公司于昨日发布了新款iPhone...其股价上涨了2%..."
```

这个字符串对人类友好，但对程序不友好。如果我们想从这段文本中提取出"公司名"和"股价变化"并用在后续逻辑中，则需要编写复杂且容易出错的解析代码。

聊天模型的 `with_structured_output()` 方法允许我们预先定义一个期望的数据结构，并要求大模型必须按照这个结构返回信息。

### with_structured_output() 方法

这是获得结构化输出的最简单、最可靠的方法。该方法将**输出结构**作为参数输入，返回一个类似 model 的 `Runnable`。不同之处在于执行 Runnable 后的输出结果，输出的不是字符串或消息，而是输出与给定输出结构相对应的对象。

**基本用法：**
```python
# 1. 定义输出结构
schema = {"foo": "bar"}
# 2. 绑定schema，生成支持结构化返回的 Runnable 实例
model_with_structure = model.with_structured_output(schema)
# 3. 执行
structured_output = model_with_structure.invoke(user_input)
```

**方法定义：**
```python
with_structured_output(
    schema: dict[str, Any] | type[_BM] | type | None = None,
    *,
    method: Literal['function_calling', 'json_mode', 'json_schema'] = 'json_schema',
    include_raw: bool = False,
    strict: bool | None = None,
    **kwargs: Any,
) → Runnable
```

**关键参数说明：**

| 参数 | 说明 |
|------|------|
| `schema` | 输出结构，可以传入 JSON、TypedDict、Pydantic、OpenAI 函数/工具 |
| `method` | LLM 的生成方法：`json_schema`（默认）、`function_calling`、`json_mode` |
| `include_raw` | `True` 返回包含 `raw`、`parsed`、`parsing_error` 的字典 |
| `strict` | `True` 保证模型输出与 Schema 完全匹配 |

**返回值：**
- 如果 `schema` 是 Pydantic 类，则 Runnable 输出 Pydantic 对象
- 否则 Runnable 输出一个字典

### 输出格式示例

#### 返回 Pydantic 对象

```python
from langchain_openai import ChatOpenAI
from typing import Optional
from pydantic import BaseModel, Field

model = ChatOpenAI(model="gpt-4o-mini")

class WeatherInfo(BaseModel):
    """查询天气信息。"""
    city: str = Field(description="城市名称")
    temperature: str = Field(description="当前温度")
    condition: str = Field(description="天气状况")
    humidity: Optional[int] = Field(default=None, description="湿度百分比")

structured_model = model.with_structured_output(WeatherInfo)
result = structured_model.invoke("查询北京今天的天气")
print(result)
# 输出：city='北京' temperature='22°C' condition='晴朗' humidity=45
```

**支持嵌套输出：**
```python
class Data(BaseModel):
    """获取多个城市的天气数据。"""
    weather_list: List[WeatherInfo]

structured_model = model.with_structured_output(Data)
result = structured_model.invoke("查询北京和上海今天的天气")
# 输出：weather_list=[WeatherInfo(city='北京', ...), WeatherInfo(city='上海', ...)]
```

#### 返回 TypedDict

TypedDict 用于为字典对象提供精确的、结构化的类型提示：

```python
from typing_extensions import Annotated, TypedDict

class WeatherInfo(TypedDict):
    """查询天气信息。"""
    city: Annotated[str, ..., "城市名称"]
    temperature: Annotated[str, ..., "当前温度"]
    condition: Annotated[str, ..., "天气状况"]
    humidity: Annotated[Optional[int], None, "湿度百分比"]

structured_model = model.with_structured_output(WeatherInfo)
result = structured_model.invoke("查询北京今天的天气")
# 输出：{'city': '北京', 'temperature': '22°C', 'condition': '晴朗', 'humidity': 45}
```

#### 返回 JSON

需要定义 JSON Schema：

```python
json_schema = {
    "title": "weather_info",
    "description": "查询天气信息。",
    "type": "object",
    "properties": {
        "city": {"type": "string", "description": "城市名称"},
        "temperature": {"type": "string", "description": "当前温度"},
        "condition": {"type": "string", "description": "天气状况"},
        "humidity": {"type": "integer", "description": "湿度百分比", "default": None},
    },
    "required": ["city", "temperature", "condition"],
}

structured_model = model.with_structured_output(json_schema)
result = structured_model.invoke("查询北京今天的天气")
# 输出：{'city': '北京', 'temperature': '22°C', 'condition': '晴朗', 'humidity': 45}
```

### 选择输出格式

使用 Union 类型创建联合类型属性的父模式，让模型根据输入选择不同的输出格式：

```python
from pydantic import BaseModel, Field
from typing import Union

class WeatherInfo(BaseModel):
    """查询天气信息。"""
    city: str = Field(description="城市名称")
    temperature: str = Field(description="当前温度")
    condition: str = Field(description="天气状况")
    humidity: Optional[int] = Field(default=None, description="湿度百分比")

class ConversationalResponse(BaseModel):
    """以对话的方式回应。待人友善，乐于助人。"""
    response: str = Field(description="对用户查询的会话响应")

class FinalResponse(BaseModel):
    final_output: Union[WeatherInfo, ConversationalResponse]

structured_model = model.with_structured_output(FinalResponse)

result = structured_model.invoke("查询北京今天的天气")
# 输出：final_output=WeatherInfo(city='北京', temperature='22°C', condition='晴朗', humidity=45)

result = structured_model.invoke("你好")
# 输出：final_output=ConversationalResponse(response='你好！有什么我可以帮助你的吗？')
```

### 实用场景

#### 场景1：作为信息提取器

```python
from langchain_openai import ChatOpenAI
from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

model = ChatOpenAI(model="gpt-4o-mini")

class Person(BaseModel):
    """一个人的信息。"""
    # 注意：每个字段都是 Optional —— 允许 LLM 在不知道答案时输出 None
    # 每个字段都有 description —— LLM 使用这个描述，好的描述可以提高提取结果
    name: Optional[str] = Field(default=None, description="这个人的名字")
    hair_color: Optional[str] = Field(default=None, description="如果知道这个人头发的颜色")
    skin_color: Optional[str] = Field(default=None, description="如果知道这个人的肤色")
    height_in_meters: Optional[str] = Field(default=None, description="以米为单位的高度")

structured_model = model.with_structured_output(schema=Person)
messages = [
    SystemMessage(content="你是一个提取信息的专家，只从文本中提取相关信息。如果您不知道要提取的属性的值，属性值返回null"),
    HumanMessage(content="史密斯身高6英尺，金发。")
]
result = structured_model.invoke(messages)
print(result)
# 输出：name='史密斯' hair_color='金发' skin_color=None height_in_meters='1.83'
```

#### 场景2：与工具结合使用

**注意：** 使用聊天模型原生的工具搭配结构化输出并不好用！更好的用法见 LangGraph Agent 能力。

**方式1：使用 `with_structured_output()` 的 `tools` 参数**

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.tools import tool

model = ChatOpenAI(model="gpt-4o-mini")

class SearchResult(BaseModel):
    """结构化搜索结果。"""
    query: str = Field(description="搜索查询")
    findings: str = Field(description="调查结果摘要")

@tool
def web_search(query: str) -> str:
    """在网上搜索信息。"""
    return "西安今天多云转小雨，气温18-23度"

structured_search_model = model.with_structured_output(
    SearchResult,
    tools=[web_search],
    strict=True,
    include_raw=True,
)
result = structured_search_model.invoke("搜索当前最新的西安的天气")
# 注意：这只会返回工具调用请求，不会自动执行工具
```

**方式2：拆解能力，单独依次执行（推荐）**

当需要同时使用结构化输出和其他工具时，需要注意顺序：
1. 首先绑定工具
2. 其次添加结构化输出

```python
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

model = ChatOpenAI(model="gpt-4o-mini")

class SearchResult(BaseModel):
    """结构化搜索结果。"""
    query: str = Field(description="搜索查询")
    findings: str = Field(description="调查结果摘要")

@tool
def web_search(query: str) -> str:
    """在网上搜索信息。"""
    return "西安今天多云转小雨，气温18-23度，东南风2级，空气质量良好"

# 第一次调用：绑定工具，获取工具调用
model_with_search = model.bind_tools([web_search])
messages = [HumanMessage("搜索当前最新的西安的天气")]
ai_msg = model_with_search.invoke(messages)
messages.append(ai_msg)

# 执行工具调用，将结果加入消息列表
for tool_call in ai_msg.tool_calls:
    tool_msg = web_search.invoke(tool_call)
    messages.append(tool_msg)

# 第二次调用：使用结构化输出处理包含工具结果的消息
structured_search_model = model_with_search.with_structured_output(SearchResult)
result = structured_search_model.invoke(messages)
print(result)
# 输出：query='西安天气' findings='西安今天多云转小雨，气温18-23度，东南风2级，空气质量良好。'
```

**流程总结：**
```
第一次调用：HumanMessage → 模型 → AIMessage（含工具调用请求）
                                            ↓
                                    执行工具 → ToolMessage
                                            ↓
第二次调用：HumanMessage + AIMessage + ToolMessage → 结构化输出模型 → Pydantic对象
```


