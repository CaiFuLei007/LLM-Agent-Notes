

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





