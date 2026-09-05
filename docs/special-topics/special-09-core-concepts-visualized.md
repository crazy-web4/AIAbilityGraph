# 番外篇 Special-09：核心概念图解 —— 20 张图讲透大模型

> 🖼️ **一图胜千言** —— 本篇用 20 张示意图把大模型最容易"似懂非懂"的核心概念画出来，每张图配极简讲解。适合：初学建立直觉、复习串讲、面试前热身。概念定义见 [术语表](../GLOSSARY.md)，本篇只做"可视化理解"。

---

## 学习目标

- [ ] 在脑中建立"文本 → token → 向量 → 注意力 → 概率 → 生成"的完整数据流
- [ ] 用图解释 Transformer、KV Cache、MoE、LoRA、量化、RLHF 等机制
- [ ] 用图讲清 RAG、Agent Loop、MCP、模型路由等系统如何运转

---

## 图 1：文本如何变成模型输入（Tokenization）

```mermaid
flowchart LR
    T["文本：'我爱 DeepSeek'"] --> TOK["分词器 Tokenizer<br/>(BPE/SentencePiece)"]
    TOK --> IDS["token id 序列<br/>例如: 我(5290) 爱(9862)<br/>Deep(2132) Seek(7741)"]
    IDS --> EMB["Embedding 查表<br/>每个 id → 一个向量"]
    EMB --> VEC["向量矩阵<br/>(序列长度 × 隐藏维度)"]
```

**直觉**：模型不认识字，只认识数字。分词器把文本切成 subword（兼顾生僻词与词表大小），再查 embedding 表变成向量。token 也是计费和上下文长度的单位。

---

## 图 2：Embedding 向量空间（语义 = 位置）

```mermaid
flowchart LR
    subgraph Space["高维向量空间（示意 2D）"]
        A["🐱 '猫'"] --- B["🐈 '小猫'"]
        D["🐶 '狗'"] --- E["🐕 '小狗'"]
        A -. "语义距离近" .- B
        D -. "语义距离近" .- E
        A -. "语义距离远" .- D
        F["📱 '手机'"] --- G["📲 '电话'"]
    end
    Q["query 向量"] --> SIM["计算余弦相似度<br/>方向越一致越相似"]
    SIM --> TOP["取最近的 k 个<br/>(RAG 检索 / KNN)"]
```

**直觉**：embedding 把"语义"编码成向量方向——语义近的词/句子在空间里聚在一起。RAG 检索就是在这个空间里找"离问题最近的知识"。

---

## 图 3：一个 Transformer Block（大模型的"千层"重复单元）

```mermaid
flowchart TB
    X["输入向量 x"] --> N1["LayerNorm"]
    N1 --> ATT["多头自注意力<br/>让每个 token 看其他 token"]
    ATT --> ADD1["残差相加: x + 注意力输出"]
    ADD1 --> N2["LayerNorm"]
    N2 --> FFN["前馈网络 FFN<br/>(两层 MLP + 激活)<br/>模型知识主要存这里"]
    FFN --> ADD2["残差相加"]
    ADD2 --> Y["输出向量 → 下一个 block"]
```

**直觉**：大模型 = 几十个这样的 block 叠起来。注意力负责"信息流动"（token 之间交换信息），FFN 负责"思考与记忆"（变换、存储知识）。残差连接让深层网络可训练。

---

## 图 4：注意力机制（Q-K-V，像查字典）

```mermaid
flowchart LR
    subgraph Token["当前 token：'它'"]
        Q["Query 我要找什么信息?"]
    end
    subgraph Others["句子里其他 token"]
        K1["Key: '猫'"] 
        K2["Key: '在'"]
        K3["Key: '睡觉'"]
        V1["Value: 🐱含义"]
        V2["Value: 在"]
        V3["Value: 😴含义"]
    end
    Q -->|"Q·K 算相似度"| SCORE["注意力分数<br/>softmax 归一化"]
    K1 & K2 & K3 --> SCORE
    SCORE -->|"加权"| V1 & V2 & V3
    V1 & V2 & V3 --> OUT["加权求和 = '它' 的新表示<br/>（大概率指向'猫'）"]
```

**直觉**：每个词发出一个 Query（我想知道什么），其他词亮出 Key（我是什么），Q·K 越匹配权重越高，再按权重取它们的 Value（实际内容）。多头 = 多组并行关注不同关系（语法/指代/语义）。

---

## 图 5：自回归生成 + KV Cache（为什么越生成越"占显存"）

```mermaid
flowchart TB
    subgraph Step1["生成第 1 个 token"]
        P1["输入 prompt(5 token)"] --> O1["算出每个位置的 K/V 并缓存"]
        O1 --> G1["输出 token 6"]
    end
    subgraph Step2["生成第 2 个 token"]
        G1 --> P2["只需算新 token 的 Q"]
        O1 -.->|"复用缓存的 K/V"| P2
        P2 --> G2["输出 token 7"]
    end
    subgraph Cache["KV Cache 随序列变长线性增长"]
        C["显存 ≈ 2 × 层数 × KV头数 × 维度 × 序列长度"]
    end
    Step2 --> Cache
```

**直觉**：没有 KV Cache，每生成一个 token 都要重算整句（浪费）；有了缓存，历史 token 的 K/V 只算一次。代价是缓存随对话变长而膨胀——这是长上下文和高并发推理的显存瓶颈，也是 GQA/MLA/量化要压缩它的原因。

---

## 图 6：大模型养成三阶段（预训练 → SFT → 对齐）

```mermaid
flowchart LR
    subgraph PT["① 预训练"]
        PT1["万亿 token 文本<br/>next-token 预测"] --> PT2["基座模型<br/>有知识但'不会聊天'"]
    end
    subgraph SFT["② 监督微调 SFT"]
        PT2 --> SFT1["数万条<br/>指令-回答 样本"]
        SFT1 --> SFT2["会听指令、<br/>按对话格式回答"]
    end
    subgraph ALI["③ 对齐 RLHF/DPO"]
        SFT2 --> AL1["人类/AI 偏好数据<br/>(哪个回答更好)"]
        AL1 --> AL2["有用·诚实·无害<br/>的 Chat 模型"]
    end
```

**直觉**：预训练让模型"读书"学知识（无标签）；SFT 教它"怎么答题"（模仿示范）；对齐教它"怎么答得让人满意、不闯祸"（偏好优化）。

---

## 图 7：RLHF vs DPO（两条对齐路线）

```mermaid
flowchart TB
    subgraph RLHF["RLHF（三步走）"]
        R1["SFT 模型"] --> R2["训练奖励模型 RM<br/>学习人类偏好打分"]
        R2 --> R3["PPO 强化学习<br/>策略生成 → RM 打分 → 更新"]
        R3 --> R4["对齐模型<br/>(强但工程复杂)"]
    end
    subgraph DPO["DPO（一步走）"]
        D1["SFT 模型 + 成对偏好数据<br/>(chosen vs rejected)"] --> D2["直接优化<br/>让 chosen 概率 ↑<br/>rejected 概率 ↓"]
        D2 --> D3["对齐模型<br/>(简单稳定)"]
    end
```

**直觉**：RLHF 先训一个"评委"（奖励模型）再用 RL 反复考试；DPO 跳过评委，直接从"好答案/坏答案"成对数据中学。工程上 DPO 简单，效果上 RLHF 上限常被认为更高，二者都常用。

---

## 图 8：LoRA（不动大模型，挂个"小补丁"）

```mermaid
flowchart TB
    subgraph Freeze["冻结的原权重 W（不训练）"]
        W["W: 大矩阵 d×d<br/>(例如 4096×4096)"]
    end
    subgraph Train["训练的低秩旁路"]
        IN["输入 x"] --> B["B: d×r"]
        B --> A["A: r×d<br/>r=8/16/32"]
        A --> ADD["ΔW = B·A"]
    end
    IN --> W
    W --> OUT0["W·x"]
    ADD --> OUT["输出 = W·x + (B·A)·x"]
    OUT0 --> OUT
```

**直觉**：微调引起的权重变化是"低秩"的（信息维度低），用两个瘦长矩阵 B·A 就能表达。可训练参数降数百倍；推理时可把 `B·A` 合并回 W，零额外延迟。QLoRA 进一步把 W 量化到 4bit 省显存。

---

## 图 9：量化（用更"粗"的数字存权重）

```mermaid
flowchart LR
    subgraph FP16["FP16 权重"]
        F["0.123456, -0.987654, ...<br/>每个数 16 bit"]
    end
    subgraph INT4["INT4 量化"]
        Q["把连续值映射到 16 个档位<br/>每个数 4 bit"]
    end
    FP16 -->|"缩放+取整"| Q
    Q -->|"显存 ≈ 1/4，吞吐 ↑"| BEN["精度略降<br/>需评测验证"]
```

**直觉**：权重不需要无限精度，把 16bit 浮点压到 8bit/4bit 整数，显存大降、速度大升，代价是微小精度损失。GPTQ/AWQ（训练后量化）、SmoothQuant（激活量化）、FP8（新硬件原生）是常见方案。

---

## 图 10：MoE（混合专家：很多专家，但每次只问几个）

```mermaid
flowchart TB
    T["一个 token 进来"] --> R["路由器 Router/Gate<br/>给每个专家打分"]
    R --> E1["专家 1"]
    R --> E2["专家 2 ✅ 选中"]
    R --> E3["专家 3 ✅ 选中"]
    R --> E4["专家 4"]
    R --> E5["专家 N..."]
    E2 & E3 --> M["加权合并 → 输出"]
    NOTE["总参数 = 所有专家(知识多)<br/>激活参数 = 被选中的 2 个(计算省)"]
```

**直觉**：像医院分诊——有很多专科医生（专家 FFN），路由器根据病人（token）决定送哪 2 位。模型总知识量大（所有专家都在显存），但每个 token 只过少数专家，计算便宜。

---

## 图 11：三种分布式并行（切数据 / 切层内 / 切层）

```mermaid
flowchart TB
    subgraph DP["数据并行 DP"]
        DP1["GPU1: 完整模型 + 数据片 A"] 
        DP2["GPU2: 完整模型 + 数据片 B"]
        DP1 -.梯度同步.-> DP2
    end
    subgraph TP["张量并行 TP（层内切矩阵）"]
        TP1["GPU1: 权重左半"] --- TP2["GPU2: 权重右半"]
        TP1 -.每层通信.-> TP2
    end
    subgraph PP["流水线并行 PP（按层切）"]
        PP1["GPU1: 第1-10层"] --> PP2["GPU2: 第11-20层"] --> PP3["GPU3: 第21-30层"]
    end
```

**直觉**：DP 是"多个人做同一套卷子的不同题"；TP 是"一层的大矩阵拆给多张卡一起算"（通信多，放同一台机器）；PP 是"流水线接力，每人负责几层"（通信少，可跨机器）。大模型训练三者组合（3D 并行）。

---

## 图 12：思维链（CoT）：把"心算"写出来

```mermaid
flowchart LR
    Q["问题: 鸡兔同笼..."] --> DIRECT["直接回答<br/>(容易错)"]
    Q --> COT["分步思考:<br/>1. 设鸡 x 兔 y<br/>2. 列方程<br/>3. 求解<br/>4. 验算"]
    COT --> ANS["答案 (正确率↑)"]
    COT -.->|"推理模型把它<br/>内化成长思考"| REASON["o1/R1 长 CoT<br/>会验算、回溯"]
```

**直觉**：模型每生成一个 token 都是一次"前向计算"，把推理步骤写出来等于给它更多串行计算预算。复杂题"多想几步"正确率显著提升；推理模型靠 RL 把这种思考行为训练成自发能力。

---

## 图 13：RAG（先查资料，再回答）

```mermaid
flowchart LR
    subgraph Offline["离线：建知识库"]
        DOC["文档"] --> CHUNK["分块 Chunk"]
        CHUNK --> EMB1["Embedding"]
        EMB1 --> VDB[("向量库")]
    end
    subgraph Online["在线：问答"]
        Q2["用户问题"] --> EMB2["Embedding"]
        EMB2 --> SEARCH["向量+关键词<br/>混合检索 top20"]
        SEARCH --> RERANK["Reranker 精排 top5"]
        VDB --> SEARCH
        RERANK --> PROMPT["问题 + 相关片段<br/>拼进 Prompt"]
        PROMPT --> LLM["大模型生成<br/>(带引用)"]
    end
```

**直觉**：开卷考试——不要求模型背下所有知识，而是先检索出相关资料再让它"照着资料答"。解决知识过时、幻觉、私有数据接入三大问题；引用来源让答案可核验。

---

## 图 14：Agent Loop（感知-思考-行动-观察的循环）

```mermaid
flowchart TB
    GOAL["用户目标"] --> THINK["LLM 思考: 下一步做什么?"]
    THINK --> DECIDE{"需要工具?"}
    DECIDE -->|需要| CALL["调用工具<br/>(搜索/代码/API/MCP)"]
    CALL --> OBS["观察结果"]
    OBS --> THINK
    DECIDE -->|信息足够| ANSWER["输出结果"]
    THINK -.->|"每一步存检查点<br/>可恢复/可审批"| CP[("状态存储")]
```

**直觉**：普通聊天是"一问一答"；Agent 是"给个目标，模型自己决定调什么工具、看结果、再决定下一步"，直到完成。工程关键：工具能力、记忆、检查点（崩溃可恢复）、预算护栏（防死循环）、人机审批（防闯祸）。

---

## 图 15：MCP（Agent 接工具的"USB-C 接口"）

```mermaid
flowchart LR
    subgraph Host["Agent 应用 (Host)"]
        AG["Agent"] --> CL["MCP Client"]
    end
    CL <-->|"标准协议<br/>Streamable HTTP + OAuth"| PS1["MCP Server: 数据库"]
    CL <-->|"标准协议"| PS2["MCP Server: CRM"]
    CL <-->|"标准协议"| PS3["MCP Server: 文件系统"]
    REG["MCP Registry<br/>(发现/签名)"] -.-> PS1
    REG -.-> PS2
```

**直觉**：没有 MCP，每接一个工具都要写一套定制对接（N×M 噩梦）；有了 MCP，任何 Agent 都能即插即用任何 MCP Server，就像 USB 统一了外设接口。MCP 管"Agent 连工具"，A2A 管"Agent 连 Agent"。

---

## 图 16：模型级联路由（便宜模型先上，难题升级）

```mermaid
flowchart TD
    REQ["请求"] --> SMALL["小/快/便宜模型"]
    SMALL --> EASY{"置信度/难度判断"}
    EASY -->|简单、有把握| OUT1["直接返回（占 70-80%）"]
    EASY -->|困难/低置信| MID["中等模型"]
    MID --> E2{"还能解决?"}
    E2 -->|是| OUT2["返回"]
    E2 -->|复杂推理| BIG["推理模型/最强模型"]
    BIG --> OUT3["返回"]
    CACHE["缓存命中"] -.-> OUT1
```

**直觉**：像医院分级诊疗——小病在社区（小模型，几分钱），大病转专科（大模型），疑难杂症请专家（推理模型）。平均成本可降 5–20 倍而质量几乎不变。配合缓存、prompt caching 效果叠加。

---

## 图 17：vLLM 连续批处理（GPU 像餐厅翻台）

```mermaid
flowchart LR
    subgraph Static["静态批处理（慢）"]
        S1["请求A ████████"] 
        S2["请求B █"]
        S3["请求C █████"]
        S4["必须等 A/B/C 全部结束<br/>才接待下一批"]
    end
    subgraph Cont["连续批处理（快）"]
        C1["A ████████"]
        C2["B █ 完成立即换 D"]
        C3["C █████ 完成立即换 E"]
        C4["逐 token 动态进出<br/>GPU 不空闲"]
    end
```

**直觉**：静态批处理像包车，要等所有人到齐才走；连续批处理（continuous batching）像餐厅翻台，谁吃完谁走、立刻补新客人。配合 PagedAttention 像操作系统一样分页管理 KV Cache 显存，吞吐提升数倍。

---

## 图 18：智能体评估（不只看"答得对不对"，看"任务完没完成"）

```mermaid
flowchart LR
    TASK["评测任务集<br/>(真实+对抗+边界)"] --> RUN["Agent 在沙箱执行"]
    RUN --> RES["产出: 结果 + 完整轨迹"]
    RES --> J1["终态断言<br/>目标达成? (对答案/查状态)"]
    RES --> J2["轨迹评估<br/>工具选对? 参数对?<br/>步数/成本?"]
    RES --> J3["安全评估<br/>越权? 被注入?"]
    J1 & J2 & J3 --> GATE{"发布门禁"}
    GATE -->|达标| SHIP["灰度发布"]
    GATE -->|不达标| BLOCK["阻断 + badcase 回流"]
```

**直觉**：传统模型评"答题分"；Agent 要在沙箱里真跑，评"目标达成没、工具用对没、有没有闯祸"。Eval 是 Agent 的单元测试，prompt/工具变更都要过门禁。

---

## 图 19：HITL 与持久化执行（等人审批，但不占着机器）

```mermaid
flowchart TB
    RUN["Agent 执行中"] --> RISK{"高危动作?<br/>(删/发/付款)"}
    RISK -->|低风险| AUTO["自动执行"]
    RISK -->|高风险| SAVE["保存检查点 → 挂起<br/>Worker 释放"]
    SAVE --> NOTIFY["发审批卡片给人"]
    NOTIFY --> WAIT["等待...(可等数小时)"]
    WAIT --> DECISION{"人决定"}
    DECISION -->|批准| RESUME["加载检查点 → 继续"]
    DECISION -->|拒绝| STOP["终止/换路"]
```

**直觉**：人可能 3 小时后才审批，系统不能干等。把任务状态存下来、释放计算资源，审批结果回来再"断点续跑"。这就是 Durable Execution——Agent 可随时崩溃、暂停、恢复。

---

## 图 20：测试时计算光谱（多想的 N 种方式）

```mermaid
flowchart LR
    A["直接答<br/>×1"] --> B["CoT 分步<br/>×1.5"]
    B --> C["自洽性<br/>采样n次投票 ×n"]
    C --> D["Best-of-N<br/>n次+验证器选最佳"]
    D --> E["长 CoT 推理模型<br/>×5–50"]
    E --> F["树搜索 MCTS<br/>×几十–几百"]
```

**直觉**："想得更久"有两种维度——**宽度**（多想几个方案再选：自洽性/Best-of-N/树搜索）和**深度**（一条路想到底、中途验算回溯：长 CoT）。越往后越强也越贵，按任务难度选择，是推理模型时代的成本-质量调度核心。

---

## 小结：一张图串起整个大模型数据流

```mermaid
flowchart LR
    TXT["文本"] -->|分词/embedding| TRANS["Transformer 堆叠<br/>(注意力+FFN)"]
    TRANS -->|预训练+SFT+对齐| MODEL["大模型"]
    MODEL -->|API| APP["应用"]
    APP -->|知识接地| RAG["RAG 检索"]
    APP -->|采取行动| AGENT["Agent + 工具(MCP)"]
    APP -->|按难度调度| ROUTE["模型路由<br/>小模型→推理模型"]
    APP -->|质量保证| EVAL["评测 + HITL + 护栏"]
    RAG & AGENT --> INFRA["推理服务<br/>vLLM/KV缓存/量化"]
```

记住这条主线：**文本变向量 → Transformer 算注意力 → 预训练学知识、对齐学做人 → 应用层靠 RAG 接地、靠 Agent 行动、靠路由省钱、靠评测和护栏兜底**。

---

## 延伸阅读

- 每个概念的完整定义：[术语表](../GLOSSARY.md)
- 平台与系统全景：[Special-06 智能体平台架构](special-06-agent-platform-architecture.md)
- 图 20 深入：[Special-07 推理模型与测试时计算](special-07-reasoning-models.md)
- 技术选型落地：[Special-03 技术选型手册](special-03-tech-selection.md)

---

[← 返回番外篇目录](README.md) | [返回教程主目录](../../README.md)
