# service-mind-Agent :企业级电商客服Agent系统

### 为什么选电商客服？

电商客服是 Agent 最经典的落地场景之一：业务逻辑清晰（查订单、退换货、推荐商品、售后处理），大家容易理解，面试中也经常被问到。做完这个项目，不仅能掌握 Agent 核心技术栈，还能直接写进简历。


### 技术演进路线

本项目会按照由浅入深的节奏，逐步叠加 Agent 相关技术：

**基础篇**
- 纯 Prompt 实现客服对话
- 结构化输出（Structured Output）
- 多轮对话管理

**进阶篇**
- ReAct 范式的 Agent（思考-行动交替，最经典的 Agent 范式）
- 工具调用 / Function Calling（查订单、查库存等）
- MCP（Model Context Protocol）集成
- RAG 检索增强生成（接入商品库、FAQ、退换货政策等）

**高级篇**
- Multi-Agent 协作（客服路由、售前售后分流）✅
- Memory：短期记忆 & 长期记忆✅
- Skill：可复用的能力模块（退货处理、订单跟踪等标准化流程）✅
- Agent 评估体系 ✅

**生产篇**
- Guardrails 安全护栏（Prompt Injection 检测、输出幻觉校验、敏感信息过滤、意图越界拦截） ✅
- Human-in-the-Loop 人机协作（置信度评估与自动转人工、Agent↔真人客服交接协议、上下文传递） ✅
- Agent Observability 可观测性（调用链 Trace、Token/延迟指标采集、工具成功率看板、异常告警） ✅

---

## 项目架构 & 更新历史

> 这是本项目最核心的部分，会随着每一期的更新持续完善。

### 当前架构

```
service-mind/
├── main.py                # 入口（支持单 Agent / Multi-Agent 模式切换 + memory/skills 命令）
├── requirements.txt
├── .env.example
├── pyproject.toml                 # 项目配置
├── docker-compose.yml             # 本地开发环境（PostgreSQL + Redis）
├── demo_guard.py           		# SkillGuard 演示脚本
│
├── app/                           # 主 Bot 全部代码 + 数据
│   ├── main.py                    # FastAPI 入口（SSE / 安全护栏 / 限流 / 认证）
│   ├── auth.py                    # JWT 认证 + bcrypt 密码哈希
│   ├── database.py                # 异步数据库引擎（PostgreSQL + asyncpg）
│   ├── limiter.py                 # Redis 限流
│   ├── models.py                  # SQLAlchemy ORM 模型（5 张表）
│   ├── repository.py              # Repository 模式（Mock / SQLAlchemy 双实现）
│   ├── config/
│   │   └── settings.py            # 配置管理（从 .env 读取，含 MCP / RAG / Multi-Agent / 														Memory / Skill / Evaluation 配置）
│   ├── core/                      # 核心基础设施层
│   │   ├── llm.py                 # 全局共享 OpenAI 客户端连接池
│   │   ├── embedding.py           # 全局共享 Embedding 客户端
│   │   ├── logger.py              # 结构化日志（structlog）
│   │   ├── guardrails.py          # 安全护栏（Prompt Injection 检测 + 输出脱敏）
│   │   ├── observability.py       # 可观测性 Trace 装饰器
│   │   ├── breaker.py             # LLM API 熔断器
│   │   ├── events.py              # 全局事件总线（发布/订阅）
│   │   └── semantic_router.py     # 语义路由（Embedding + 余弦相似度）
│   ├── prompts/
│   │   ├── customer_service.py    # 电商客服 system prompt（含工具使用指南 + 记忆能力）
│   │   ├── summarizer.py          # 历史摘要 prompt
│   │   ├── agents.py              # Multi-Agent 子 Agent prompt（售前/售后/投诉 + Router）
│   │   ├── memory.py              # 记忆提取 prompt（短期 STM / 长期 LTM 事实抽取）
│   │   └── evaluation.py          # LLM-as-judge prompt（回答质量 / 幻觉 / 过程合理性）
│   ├── routers/
│   │   └── auth.py                # 注册 / 登录 API
│   ├── schemas/
│   │   └── response.py            # 结构化输出 schema（Pydantic）
│   │   └── state.py               # 图引擎全局状态 AgentState
│   ├── agent/                     # Agent 核心实现 + 全部 Agent 技术栈（tools / rag / skills）
│   │   ├── chat.py                # 核心 ReAct 循环（集成 MemoryManager + SkillManager）
│   │   ├── summarizer.py          # LLM 自我压缩老对话（支持工具消息）
│   │   ├── storage.py             # 会话 JSON 持久化（含短期记忆）
│   │   ├── storage_v2.py          # 会话存储 v2（aiofiles 异步 I/O）
│   │   ├── memory/                # 记忆系统（第7期）
│   │   │   ├── __init__.py        # 导出 MemoryManager / ShortTermMemory / LongTermMemory
│   │   │   ├── manager.py         # MemoryManager：统一管理短期 + 长期记忆
│   │   │   ├── short_term.py      # 短期记忆：会话内事实提取
│   │   │   ├── long_term.py       # 长期记忆：跨会话持久化（JSON per user）
│   │   │   └── extractor.py   		# 统一记忆提取器（STM + Profile + LTM 三层一体）
│   │   │   ├── profile_memory.py   # 用户画像记忆（PostgreSQL upsert）
│   │   │   ├── vector_memory.py    # 向量记忆（pgvector 语义检索）
│   │   ├── skills/                # Skill 模块（第8期）：代码 + 技能内容分层
│   │   │   ├── __init__.py        # 导出 SkillManager / SkillMeta
│   │   │   ├── loader.py          # SkillManager：扫描、发现、加载 SKILL.md（渐进式披露）
│   │   │   ├── guard.py           # SkillGuard 工具调用硬拦截器 
│   │   │   ├── state.py          # SkillState 运行时状态追踪  
│   │   │   └── definitions/       # 技能内容（遵循 Agent Skills 开放标准，每个一个 SKILL.md）
│   │   │       ├── process-return/
│   │   │       │   └── SKILL.md   # 退货退款处理技能（确认订单→校验资格→退款→告知进度）
│   │   │       ├── track-order/
│   │   │       │   └── SKILL.md   # 订单物流跟踪技能（查单→查物流→综合建议）
│   │   │       └── product-recommend/
│   │   │           └── SKILL.md   # 商品推荐技能（了解需求→查偏好→搜索→推荐）
│   │   │       └── member-benefit
│   │   │           └── SKILL.md   # 会员权益查询
│   │   ├── strategies/            # (upcoming) Agent 执行策略
│   │   ├── tools/                 # 电商工具集（Function Calling）
│   │   │   ├── mock_data.py       # Mock 数据：订单、商品、物流
│   │   │   ├── registry.py        # 本地工具注册表 + OpenAI schema + 分发执行
│   │   │   ├── manager.py         # ToolManager：统一管理本地 + MCP 工具（支持 allowed_tools 																					过滤）
│   │   │   ├── user_orders.py      # list_user_orders 用户订单概要列表   
│   │   │   ├── order.py           # 查询订单详情
│   │   │   ├── product.py         # 搜索商品信息
│   │   │   ├── logistics.py       # 查询物流轨迹
│   │   │   ├── refund.py          # 申请退款
│   │   │   ├── knowledge.py       # search_knowledge：RAG 政策/FAQ 检索
│   │   │   ├── memory_tool.py     # recall_user_memory：查询用户记忆
│   │   │   └── skill_tool.py      # load_skill：按需加载技能指令
│   │   └── rag/                   # RAG 模块
│   │       ├── chunker.py         # Markdown → Chunk（按二级标题切分）
│   │       ├── embedder.py        # OpenAI Embeddings 封装
│   │       ├── retriever.py       # KnowledgeRetriever：query → 向量检索
│   │       ├── backends/          # 向量后端（可切换）
│   │       │   ├── base.py        # VectorBackend 抽象接口
│   │       │   ├── numpy_backend.py   # 手写余弦 + JSON（教学透明，零依赖）
│   │       │   └── chroma_backend.py  # Chroma 嵌入式向量数据库（生产代表）
│   │       └── knowledge/         # 知识库源文档（markdown，RAG 数据源）
│   │           ├── 退换货政策.md
│   │           ├── 配送说明.md
│   │           ├── 会员权益.md
│   │           └── 常见问题FAQ.md
│   ├── mcp_client/                # MCP Client（同步封装）
│   │   ├── client.py              # MCPClient：后台线程管理异步连接
│   │   └── converter.py           # MCP Tool schema → OpenAI function calling 格式
│   ├── evaluation/                # Agent 评估体系（第9期）
│   │   ├── __init__.py            # 导出 EvalCase / Sandbox / Evaluator / RunTrace 等
│   │   ├── dataset.py             # EvalCase 数据结构 + load_dataset
│   │   ├── trace.py               # RunTrace：沙箱采集的过程+结果载体
│   │   ├── sandbox.py             # Sandbox：隔离环境 + 共享 client 插桩 + 采集
│   │   ├── metrics.py             # 过程/结果双层指标（代码规则 + LLM judge）
│   │   ├── evaluator.py           # Evaluator：跑用例 → 双层评分 → 聚合报告
│   │   └── cases.json             # 黄金测试集（~10 条，引用 mock 数据）
│   ├── multi_agent/               # Multi-Agent 协作（第6期）
│   │   ├── router.py              # 意图路由器（LLM 分类 → 子 Agent）
│   │   ├── agents.py              # SubAgent 子 Agent 类 + 配置
│   │   └── orchestrator.py        # 编排器：路由 → 执行 → 结构化提取（集成 MemoryManager + SkillManager）
│   ├── scripts/
│   │   ├── build_kb_index.py      # 离线构建知识库索引（--backend numpy/chroma）
│   │   └── run_eval.py            # 离线运行评估（--mode single/multi · --judge/--no-judge · --output）
│   └── sessions/                  # 运行时生成，已 .gitignore
│       ├── session.json           # 当前会话快照
│       ├── kb_index.json          # NumpyBackend 索引
│       ├── chroma/                # ChromaBackend 持久化目录
│       └── memory/                # 长期记忆存储（按 user_id 分文件）
│           └── {user_id}.json
│
├── mcp_server/                    # MCP Server（独立微服务）
│   └── server.py                  # FastMCP + Streamable HTTP，暴露电商工具
│
├── scripts/                       # 工具脚本
│   ├── init_db.py                 # 初始化数据库表结构
│   ├── generate_mock_data.py      # 生成 Mock 订单数据
│   ├── generate_cases.py          # 生成评估测试用例
│   └── align_test_cases.py        # 对齐测试用例
└── tests/                         # 全部测试（46 个测试文件，覆盖所有模块）
    ├── test_agent.py              # 结构化输出 + 多轮 + reset
    ├── test_conversation_management.py  # 多轮对话管理
    ├── test_react_agent.py        # ReAct Agent + Function Calling
    ├── test_mcp.py                # MCP 集成
    ├── test_rag.py                # RAG 知识库检索
    ├── test_multi_agent.py        # Multi-Agent 协作
    ├── test_memory.py             # Memory 短期记忆 & 长期记忆
    ├── test_skills.py             # Skill 可复用能力模块
    └── test_evaluation.py         # Agent 评估体系（沙箱 + 双层测评）
    新增：test_api.py、test_auth.py、test_guardrails.py、test_observability.py、test_semantic_router.py、test_rate_limit.py、test_repository.py、test_resilience.py、test_security.py、test_skill_guard.py、test_skill_state.py、test_memory_integration.py、test_hitl.py、test_graph_engine.py、test_stm.py、test_isolation.py 等
    
```

### 更新日志

| 期数 | 主题 | Tag | 日期 |
|------|------|-----|------|
| 第 1 期 | 项目框架 + 纯 Prompt 客服 + 结构化输出 | v1-prompt-and-structured-output | 2026-04-14 |
| 第 2 期 | 多轮对话管理：Summary 压缩 + JSON 持久化 | v2-conversation-management | 2026-04-18 |
| 第 3 期 | ReAct Agent + 工具调用 (Function Calling) | v3-react-and-function-calling | 2026-04-27 |
| 第 4 期 | MCP 集成 (Streamable HTTP) | v4-mcp-integration | 2026-05-01 |
| 第 5 期 | RAG 检索增强生成（FAQ + 政策知识库） | v5-rag | 2026-05-13 |
| 第 6 期 | Multi-Agent 协作（客服路由 + 售前/售后/投诉分流） | v7-multi-agent | 2026-05-17 |
| 第 7 期 | Memory：短期记忆 & 长期记忆 | v8-memory | 2026-05-23 |
| 第 8 期 | Skill：可复用能力模块（基于 Agent Skills 开放标准） | v9-skills | 2026-05-31 |
| 第 9 期 | Agent 评估体系（沙箱重跑测试集 + 过程/结果双层指标 + LLM judge） | v10-evaluation | 2026-06-06 |
| 第 10 期 | 生产化基础设施：PostgreSQL + JWT 认证 + 安全护栏 + 熔断器 + 限流 + 可观测性 + 语义路由 | v11-production-infra | 2026-07-xx |
| 第 11 期 | 记忆系统增强：统一提取器 + 画像记忆 + 向量记忆（pgvector） | v12-memory-v2 | 2026-07-xx |
| 第 12 期 | Skill 安全增强：SkillGuard 硬拦截 + SkillState 运行时追踪 + 会员权益技能 | v13-skill-guard | 2026-08-xx |



## 基础设施

本项目在第 10 期后引入了生产级基础设施，支撑 Agent 系统的稳定运行：

| 组件      | 技术选型                 | 用途                                                   |
| --------- | ------------------------ | ------------------------------------------------------ |
| 数据库    | PostgreSQL 16 + pgvector | 订单存储、用户画像、向量记忆                           |
| 缓存/限流 | Redis 7                  | 接口限流（每用户每分钟 5 次）                          |
| 认证      | JWT (HS256) + bcrypt     | 用户注册/登录/鉴权                                     |
| 日志      | structlog                | 开发环境彩色输出 / 生产环境 JSON                       |
| 熔断      | aiocircuitbreaker        | LLM API 连续失败 3 次 → 30s 快速失败                   |
| 安全护栏  | 正则 + 关键词匹配        | 输入层 Prompt Injection 检测 + 输出层手机号/身份证脱敏 |

### 本地启动

```bash
# 1. 启动基础设施
docker-compose up -d

# 2. 初始化数据库表
python scripts/init_db.py

# 3. 生成测试数据
python scripts/generate_mock_data.py

# 4. 启动服务
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
