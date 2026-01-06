# RLM 项目中文指南

> 递归语言模型（Recursive Language Models）推理引擎

本文档为开发者提供 RLM 项目的整体架构、核心逻辑和部署指南。

---

## 目录

- [项目概述](#项目概述)
- [目录结构](#目录结构)
- [核心模块详解](#核心模块详解)
- [执行流程](#执行流程)
- [部署与配置](#部署与配置)
- [代码位置索引](#代码位置索引)

---

## 项目概述

RLM（Recursive Language Models）是一个任务无关的语言模型推理范式，通过让语言模型能够**程序化地**检查、分解并递归调用自身来处理近乎无限长度的上下文。

### 核心思想

传统的 LLM 调用方式：
```python
llm.completion(prompt, model)
```

RLM 的调用方式：
```python
rlm.completion(prompt, model)
```

RLM 将上下文卸载到 REPL 环境中作为变量，让语言模型可以通过代码与上下文交互，并在 REPL 内部发起子语言模型调用。

### 相关资源

- **论文**: [arXiv:2512.24601](https://arxiv.org/abs/2512.24601)
- **博客**: [RLM Blogpost](https://alexzhang13.github.io/blog/2025/rlm/)
- **文档站点**: [RLM Documentation](https://alexzhang13.github.io/rlm/)

---

## 目录结构

```
rlm/
├── rlm/                          # 主包（约 3,860 行代码）
│   ├── __init__.py               # 包入口，导出 RLM 类
│   │
│   ├── core/                     # 核心逻辑模块
│   │   ├── rlm.py               # RLM 主类（294 行）
│   │   ├── types.py             # 数据类型定义（257 行）
│   │   ├── lm_handler.py        # TCP 套接字服务器（185 行）
│   │   └── comms_utils.py       # 通信协议工具（254 行）
│   │
│   ├── clients/                  # 语言模型客户端实现
│   │   ├── base_lm.py           # 抽象基类 BaseLM
│   │   ├── openai.py            # OpenAI 客户端（126 行）
│   │   ├── anthropic.py         # Anthropic 客户端（112 行）
│   │   ├── azure_openai.py      # Azure OpenAI 客户端（139 行）
│   │   ├── gemini.py            # Google Gemini 客户端（162 行）
│   │   ├── litellm.py           # LiteLLM 路由客户端
│   │   └── portkey.py           # Portkey 路由客户端
│   │
│   ├── environments/             # 执行环境实现
│   │   ├── base_env.py          # 基类：BaseEnv, IsolatedEnv, NonIsolatedEnv
│   │   ├── local_repl.py        # 本地 Python REPL（310 行）
│   │   ├── docker_repl.py       # Docker 容器 REPL（314 行）
│   │   ├── modal_repl.py        # Modal 云沙箱 REPL（520 行）
│   │   └── prime_repl.py        # Prime 沙箱 REPL（未实现）
│   │
│   ├── logger/                   # 日志与可视化
│   │   ├── rlm_logger.py        # JSON-lines 文件日志
│   │   └── verbose.py           # Rich 控制台输出（393 行）
│   │
│   └── utils/                    # 工具函数
│       ├── parsing.py           # 代码块和答案解析（168 行）
│       ├── prompts.py           # 系统提示词构建（130 行）
│       └── rlm_utils.py         # 辅助工具
│
├── tests/                        # 测试套件
├── examples/                     # 示例脚本
├── docs/                         # 文档站点
├── visualizer/                   # React 轨迹可视化工具
└── pyproject.toml               # 项目配置
```

---

## 核心模块详解

### 1. RLM 主类

**位置**: `rlm/core/rlm.py`

RLM 类是整个框架的核心入口点，负责协调完整的 RLM 工作流程。

#### 主要参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `backend` | str | 语言模型后端（openai, anthropic, azure_openai, gemini 等） |
| `backend_kwargs` | dict | 后端配置（model_name, api_key 等） |
| `environment` | str | 执行环境（local, docker, modal, prime） |
| `environment_kwargs` | dict | 环境配置参数 |
| `max_depth` | int | 最大递归深度（默认为 1） |
| `max_iterations` | int | 最大迭代次数（默认为 30） |
| `logger` | RLMLogger | 可选的日志记录器 |
| `verbose` | bool | 是否启用控制台输出 |

#### 核心方法

```python
class RLM:
    def __init__(self, backend, backend_kwargs, environment, ...):
        """初始化 RLM 实例，配置后端和环境"""

    def completion(self, prompt, root_prompt=None) -> RLMChatCompletion:
        """执行 RLM 补全调用，返回最终结果"""

    def _completion_turn(self, messages) -> RLMIteration:
        """单次迭代：提示 LM -> 提取代码块 -> 执行代码"""

    def _spawn_completion_context(self):
        """创建 LMHandler 和 Environment，管理资源清理"""

    def _setup_prompt(self, prompt, query_metadata) -> list:
        """构建包含上下文元数据的系统提示"""

    def _fallback_answer(self, messages) -> str:
        """达到最大深度时生成回退答案"""
```

### 2. 语言模型处理器

**位置**: `rlm/core/lm_handler.py`

LMHandler 是一个 TCP 套接字服务器，负责路由来自主进程和环境子进程的语言模型调用。

#### 组件结构

```
LMHandler
├── start()                 # 启动 TCP 服务器
├── stop()                  # 停止服务器
├── register_client()       # 注册额外的 LM 后端
├── get_client()           # 根据模型名获取客户端
└── completion()           # 执行补全调用

ThreadingLMServer          # 多线程 TCP 服务器
└── LMRequestHandler       # 处理单个请求
    ├── _handle_single()   # 处理单个提示
    └── _handle_batched()  # 处理批量提示（异步）
```

#### 通信协议

通信使用 4 字节大端序长度前缀 + JSON 负载的格式：

```
[4 字节长度] [JSON 数据]
```

相关函数位于 `rlm/core/comms_utils.py`：
- `socket_send(sock, data)`: 发送带长度前缀的 JSON
- `socket_recv(sock)`: 接收并反序列化数据

### 3. 类型系统

**位置**: `rlm/core/types.py`

定义了所有用于跟踪 RLM 执行状态、使用量和元数据的数据类。

#### 核心类型

| 类型 | 用途 |
|------|------|
| `ClientBackend` | 字面量类型，支持的后端枚举 |
| `EnvironmentType` | 字面量类型，支持的环境枚举 |
| `RLMChatCompletion` | 单次 LM 调用记录（提示、响应、使用量、耗时） |
| `REPLResult` | 代码执行结果（stdout, stderr, locals, llm_calls） |
| `CodeBlock` | 代码块及其执行结果 |
| `RLMIteration` | 单次迭代记录（提示、响应、代码块、最终答案、耗时） |
| `ModelUsageSummary` | 单模型统计（调用次数、输入/输出 token） |
| `UsageSummary` | 跨模型聚合使用量 |
| `RLMMetadata` | 配置快照 |
| `QueryMetadata` | 输入上下文分析 |

所有类型都实现了 `.to_dict()` 和 `.from_dict()` 方法以支持 JSON 序列化。

### 4. 语言模型客户端

**位置**: `rlm/clients/`

#### 抽象基类

```python
# rlm/clients/base_lm.py
class BaseLM(ABC):
    @abstractmethod
    def completion(self, prompt: str | dict) -> str:
        """同步补全调用"""

    @abstractmethod
    async def acompletion(self, prompt: str | dict) -> str:
        """异步补全调用"""

    @abstractmethod
    def get_usage_summary(self) -> UsageSummary:
        """获取累计使用量"""

    @abstractmethod
    def get_last_usage(self) -> UsageSummary:
        """获取上次调用使用量"""
```

#### 已实现的客户端

| 客户端 | 文件 | 环境变量 | 说明 |
|--------|------|----------|------|
| OpenAIClient | `openai.py` | `OPENAI_API_KEY` | 支持 OpenAI、vLLM、OpenRouter |
| AnthropicClient | `anthropic.py` | `ANTHROPIC_API_KEY` | Anthropic Claude 系列 |
| AzureOpenAIClient | `azure_openai.py` | 多个 Azure 变量 | Azure 托管的 OpenAI |
| GeminiClient | `gemini.py` | `GOOGLE_API_KEY` | Google Gemini |
| LiteLLMClient | `litellm.py` | 取决于模型 | 100+ 模型路由 |
| PortkeyClient | `portkey.py` | `PORTKEY_API_KEY` | Portkey 平台 |

#### 客户端工厂

```python
# rlm/clients/__init__.py
def get_client(backend: str, backend_kwargs: dict) -> BaseLM:
    """根据后端名称返回对应的客户端实例"""
```

### 5. 执行环境

**位置**: `rlm/environments/`

#### 环境类型

```
BaseEnv (抽象基类)
├── NonIsolatedEnv (非隔离环境)
│   ├── LocalREPL      # 本地执行
│   └── DockerREPL     # Docker 容器执行
│
└── IsolatedEnv (隔离环境)
    ├── ModalREPL      # Modal 云沙箱
    └── PrimeREPL      # Prime 沙箱（未实现）
```

#### LocalREPL（默认环境）

**位置**: `rlm/environments/local_repl.py`

- 在同一进程中通过 Python `exec()` 执行代码
- 使用安全内建函数白名单（禁止 `eval`, `exec`, `globals`, `input`）
- 跨迭代保持命名空间
- 提供给代码的函数：
  - `context`: 加载的上下文数据
  - `llm_query(prompt, model=None)`: 子语言模型查询
  - `llm_query_batched(prompts, model=None)`: 并发子查询
  - `FINAL_VAR(variable_name)`: 返回变量作为最终答案的标记

#### DockerREPL

**位置**: `rlm/environments/docker_repl.py`

- 在 Docker 容器中执行代码
- 默认镜像：`python:3.11-slim`
- 子语言模型调用通过套接字转发回主机
- 退出时清理容器

#### ModalREPL

**位置**: `rlm/environments/modal_repl.py`

- 在 Modal 云沙箱中执行代码
- 使用 HTTP 代理模式（而非套接字）：
  - 沙箱内运行 Flask 代理服务器
  - 主机轮询代理获取待处理请求
  - 沙箱代码在 `llm_query()` 处阻塞直到收到响应
- 通过加密 HTTP 隧道传输子语言模型调用
- 使用 `dill` 序列化状态到 `/tmp/rlm_state.dill`

### 6. 解析工具

**位置**: `rlm/utils/parsing.py`

#### 核心函数

```python
def find_code_blocks(text: str) -> list[str]:
    """从 LM 响应中提取 ```repl...``` 代码块"""

def find_final_answer(text: str, environment) -> str | None:
    """
    检测最终答案标记：
    - FINAL(...): 直接返回括号内容
    - FINAL_VAR(var_name): 在环境中执行获取变量值
    """

def format_iteration(iteration: RLMIteration, max_char_length: int) -> list:
    """将迭代转换为消息列表用于下次提示，截断过长结果"""

def format_execution_result(result: REPLResult) -> str:
    """格式化 REPL 执行结果（stdout, stderr, locals）"""
```

### 7. 提示词构建

**位置**: `rlm/utils/prompts.py`

#### 系统提示词

`RLM_SYSTEM_PROMPT` 包含：
- REPL 环境说明和 `context` 变量使用方法
- `llm_query()` 和 `llm_query_batched()` 函数文档
- 大上下文分块处理策略
- 代码块格式要求（使用 ` ```repl...``` `）
- 完成标记说明（`FINAL()` 或 `FINAL_VAR()`）

#### 构建函数

```python
def build_rlm_system_prompt(system_prompt: str, query_metadata: QueryMetadata) -> list:
    """构建包含上下文元数据的系统提示消息列表"""

def build_user_prompt(root_prompt: str, iteration: int) -> str:
    """构建用户提示，根据迭代次数调整指导语"""
```

### 8. 日志系统

**位置**: `rlm/logger/`

#### RLMLogger

**文件**: `rlm/logger/rlm_logger.py`

- 写入 JSON-lines 格式（每行一个条目）
- 元数据条目：`RLMMetadata`
- 迭代条目：每轮 `RLMIteration`
- 文件命名：`{name}_{timestamp}_{uuid}.jsonl`

#### VerbosePrinter

**文件**: `rlm/logger/verbose.py`

使用 Rich 库的彩色控制台输出：
- `print_metadata()`: 打印配置信息
- `print_iteration()`: 打印迭代详情
- `print_final_answer()`: 打印最终答案
- `print_summary()`: 打印使用量摘要

---

## 执行流程

### 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         用户代码                                 │
│  rlm = RLM(backend="openai", environment="local")               │
│  result = rlm.completion(prompt, root_prompt)                   │
└────────────────┬────────────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────────────┐
│                    RLM.completion() 流程                         │
│                                                                  │
│  1. 生成补全上下文                                                │
│     ├─ 创建 LMHandler（TCP 服务器）                              │
│     ├─ 注册语言模型后端                                          │
│     └─ 创建执行环境（LocalREPL/DockerREPL/ModalREPL）           │
│                                                                  │
│  2. 构建提示词（系统提示 + 元数据）                              │
│                                                                  │
│  3. 迭代循环（最多 max_iterations 次）                          │
│     ├─ _completion_turn() {                                     │
│     │    ├─ lm_handler.completion(messages)                    │
│     │    ├─ find_code_blocks(response)                         │
│     │    └─ env.execute_code(code)                             │
│     │        └─ 代码可调用 llm_query() → LMHandler             │
│     │    }                                                      │
│     ├─ find_final_answer(response)                             │
│     ├─ format_iteration() → 更新消息历史                        │
│     └─ 如果找到 FINAL：返回 RLMChatCompletion 并退出           │
│                                                                  │
│  4. 资源清理                                                     │
│     ├─ lm_handler.stop()                                       │
│     └─ environment.cleanup()                                   │
└──────────────────────────────────────────────────────────────────┘
```

### 详细执行步骤

1. **初始化阶段**
   - 用户创建 RLM 实例，配置后端和环境
   - 验证参数，设置默认值

2. **补全调用开始**
   - 调用 `completion(prompt, root_prompt)`
   - 生成上下文：创建 LMHandler 和 Environment
   - 将上下文加载到环境中

3. **迭代处理**（最多 30 次迭代）
   - 构建消息历史（系统提示 + 上下文元数据 + 历史交互）
   - 调用语言模型获取响应
   - 解析响应中的代码块（` ```repl...``` `）
   - 在环境中执行每个代码块
   - 代码可以调用 `llm_query()` 发起子语言模型查询
   - 格式化执行结果，追加到消息历史
   - 检测 `FINAL()` 或 `FINAL_VAR()` 标记

4. **终止条件**
   - 找到最终答案标记：返回结果
   - 达到最大迭代次数：生成回退答案
   - 达到最大递归深度：调用标准 LM 补全

5. **清理阶段**
   - 停止 LMHandler
   - 清理环境资源

---

## 部署与配置

### 快速安装

```bash
# 使用 uv（推荐）
curl -LsSf https://astral.sh/uv/install.sh | sh
uv init && uv venv --python 3.12
uv pip install -e .

# 或使用 pip
pip install -e .
```

### 环境变量配置

创建 `.env` 文件：

```bash
# OpenAI
OPENAI_API_KEY=sk-xxx

# Anthropic
ANTHROPIC_API_KEY=sk-ant-xxx

# Azure OpenAI
AZURE_OPENAI_API_KEY=xxx
AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Google Gemini
GOOGLE_API_KEY=xxx

# Portkey
PORTKEY_API_KEY=xxx
```

### 基础使用

```python
from rlm import RLM
from rlm.logger import RLMLogger

# 创建日志记录器（可选）
logger = RLMLogger(log_dir="./logs")

# 创建 RLM 实例
rlm = RLM(
    backend="openai",
    backend_kwargs={
        "model_name": "gpt-4o",
        "api_key": "sk-xxx",  # 或从环境变量读取
    },
    environment="local",
    environment_kwargs={},
    max_depth=1,
    max_iterations=30,
    logger=logger,
    verbose=True,
)

# 执行补全
result = rlm.completion(
    prompt="这是一段很长的文档...",
    root_prompt="请总结这篇文档的主要观点"
)

print(result.response)
```

### 使用不同环境

#### Docker 环境

```python
rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o"},
    environment="docker",
    environment_kwargs={
        "image": "python:3.11-slim",  # 可自定义镜像
    },
)
```

需要先安装 Docker：[Docker 安装指南](https://docs.docker.com/desktop/setup/install/)

#### Modal 云沙箱

```bash
# 安装 Modal
uv add modal
modal setup  # 认证账户
```

```python
rlm = RLM(
    backend="openai",
    backend_kwargs={"model_name": "gpt-4o"},
    environment="modal",
    environment_kwargs={},
)
```

### 使用不同后端

#### Anthropic Claude

```python
rlm = RLM(
    backend="anthropic",
    backend_kwargs={
        "model_name": "claude-sonnet-4-20250514",
        "max_tokens": 32768,
    },
)
```

#### Azure OpenAI

```python
rlm = RLM(
    backend="azure_openai",
    backend_kwargs={
        "model_name": "gpt-4o",
        "deployment_name": "your-deployment",
        "api_key": "xxx",
        "endpoint": "https://xxx.openai.azure.com/",
        "api_version": "2024-02-15-preview",
    },
)
```

#### Google Gemini

```python
rlm = RLM(
    backend="gemini",
    backend_kwargs={
        "model_name": "gemini-2.0-flash",
    },
)
```

#### 本地模型（vLLM）

```python
rlm = RLM(
    backend="openai",
    backend_kwargs={
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "base_url": "http://localhost:8000/v1",  # vLLM 服务地址
    },
)
```

### 轨迹可视化

启用日志记录后，可以使用可视化工具查看执行轨迹：

```bash
cd visualizer/
npm install
npm run dev  # 默认运行在 localhost:3001
```

在可视化工具中加载 `.jsonl` 日志文件即可查看完整的执行过程。

### 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试
uv run pytest tests/test_parsing.py

# 带覆盖率报告
uv run pytest --cov=rlm
```

### 代码风格检查

```bash
# 使用 ruff 检查和格式化
uv run ruff check .
uv run ruff format .
```

---

## 代码位置索引

### 核心类和函数

| 名称 | 位置 | 说明 |
|------|------|------|
| `RLM` | `rlm/core/rlm.py:24` | 主入口类 |
| `RLM.completion()` | `rlm/core/rlm.py:80` | 执行补全调用 |
| `RLM._completion_turn()` | `rlm/core/rlm.py:150` | 单次迭代逻辑 |
| `LMHandler` | `rlm/core/lm_handler.py:30` | TCP 服务器管理 |
| `LMRequestHandler` | `rlm/core/lm_handler.py:100` | 请求处理器 |

### 客户端

| 名称 | 位置 |
|------|------|
| `BaseLM` | `rlm/clients/base_lm.py:10` |
| `OpenAIClient` | `rlm/clients/openai.py:15` |
| `AnthropicClient` | `rlm/clients/anthropic.py:15` |
| `AzureOpenAIClient` | `rlm/clients/azure_openai.py:15` |
| `GeminiClient` | `rlm/clients/gemini.py:15` |
| `get_client()` | `rlm/clients/__init__.py:20` |

### 执行环境

| 名称 | 位置 |
|------|------|
| `BaseEnv` | `rlm/environments/base_env.py:10` |
| `LocalREPL` | `rlm/environments/local_repl.py:20` |
| `DockerREPL` | `rlm/environments/docker_repl.py:20` |
| `ModalREPL` | `rlm/environments/modal_repl.py:30` |

### 类型定义

| 名称 | 位置 |
|------|------|
| `RLMChatCompletion` | `rlm/core/types.py:30` |
| `REPLResult` | `rlm/core/types.py:60` |
| `RLMIteration` | `rlm/core/types.py:90` |
| `UsageSummary` | `rlm/core/types.py:120` |
| `LMRequest` | `rlm/core/comms_utils.py:20` |
| `LMResponse` | `rlm/core/comms_utils.py:50` |

### 工具函数

| 名称 | 位置 |
|------|------|
| `find_code_blocks()` | `rlm/utils/parsing.py:15` |
| `find_final_answer()` | `rlm/utils/parsing.py:40` |
| `format_iteration()` | `rlm/utils/parsing.py:80` |
| `build_rlm_system_prompt()` | `rlm/utils/prompts.py:60` |
| `build_user_prompt()` | `rlm/utils/prompts.py:90` |

### 日志

| 名称 | 位置 |
|------|------|
| `RLMLogger` | `rlm/logger/rlm_logger.py:10` |
| `VerbosePrinter` | `rlm/logger/verbose.py:20` |

---

## 设计模式

### 1. 依赖注入

RLM 接受 `backend`, `backend_kwargs`, `environment`, `environment_kwargs` 参数，允许灵活替换语言模型提供者和执行环境。

### 2. 资源管理器模式

`_spawn_completion_context()` 通过 try/finally 确保资源清理，LMHandler 和 Environment 始终被正确清理。

### 3. 基于套接字的进程间通信

非隔离环境使用 TCP 套接字与 LMHandler 通信，允许子进程查询主进程的语言模型。

### 4. HTTP 代理模式

隔离环境（Modal）使用沙箱内的 HTTP 代理，实现无需直接网络访问主进程的请求/响应机制。

### 5. 消息累积

消息历史随每次迭代增长，语言模型具有之前尝试和代码输出的完整上下文，促进"迭代问题解决"而非单次回答。

### 6. 安全代码执行

LocalREPL 使用受限的内建函数白名单，阻止 `eval`, `exec`, `compile` 等危险操作，但仍允许导入科学计算库。

### 7. 类型驱动序列化

所有数据结构都有 `.to_dict()` 方法用于 JSON 导出，支持日志记录、可视化和调试。

---

## 依赖项

### 核心依赖

```toml
anthropic>=0.75.0
google-genai>=1.56.0
openai>=2.14.0
portkey-ai>=2.1.0
python-dotenv>=1.2.1
requests>=2.32.5
rich>=13.0.0
```

### 可选依赖

```toml
# Modal 沙箱支持
modal>=0.73.0
dill>=0.3.7
```

### 开发依赖

```toml
ruff>=0.14.10      # 代码检查/格式化
pre-commit>=4.5.1  # Git 钩子
ty>=0.0.7          # 类型检查
pytest>=8.0.0      # 测试框架
pytest-asyncio     # 异步测试
pytest-cov         # 覆盖率
```

---

## 常见问题

### Q: 如何处理超长上下文？

RLM 的核心优势就是处理超长上下文。它会将上下文加载到 REPL 环境的 `context` 变量中，然后让语言模型通过代码分块处理、提取关键信息。

### Q: 子语言模型调用是如何工作的？

在 REPL 代码中调用 `llm_query(prompt)` 或 `llm_query_batched(prompts)` 时，请求通过 TCP 套接字（本地/Docker）或 HTTP 代理（Modal）发送到主进程的 LMHandler，由其调用配置的语言模型后端并返回结果。

### Q: 如何确定最终答案？

语言模型需要在响应中使用 `FINAL("答案内容")` 或 `FINAL_VAR(变量名)` 标记来指示最终答案。解析器会检测这些标记并提取结果。

### Q: max_depth 和 max_iterations 有什么区别？

- `max_iterations`: 单次 completion 调用中的最大迭代轮数
- `max_depth`: 递归调用的最大深度（用于 RLM 内部的子 RLM 调用场景）

---

*本文档最后更新：2026年1月*
