# 番外篇 Special-05：Codex Skills 工程实践记录

> 🛠 **用 15 个 Codex 技能实际打磨本教程项目** —— 安装、审计、修 bug、建回归检查、清洗文风，全程留痕可复现。

- 技能来源：飞书文章《Codex 最推荐的15个skill》（waytoagi 知识库）
- 安装位置：`~/.agents/skills/`（全局生效，共 13 个新技能；内置 3 个在 `~/.codex/skills/.system/`）
- 本文作用：归档每个技能在本项目上的真实用法、命令、改动与证据；不适用的技能记录前置条件，等条件满足时直接照做

---

## 📋 目录

1. [技能应用总览](#技能应用总览)
2. [本次实际产生的改动](#本次实际产生的改动)
3. [技能逐一带过](#技能逐一带过)
4. [复现命令清单](#复现命令清单)
5. [遗留 backlog 与后续触发场景](#遗留-backlog-与后续触发场景)

---

## 技能应用总览

| # | 技能 | 来源 | 本次在本项目的用法 | 状态 |
|---|------|------|--------------------|------|
| 1 | skill-creator | 内置 | 本次未创建新技能 | ⚪ 待用 |
| 2 | skill-installer | 内置 | 从 openai/skills 与 GitHub 安装全部 13 个技能 | ✅ 已用 |
| 3 | plugin-creator | 内置 | 本次未打包插件 | ⚪ 待用 |
| 4 | create-plan | 社区 | 先审计项目结构再产出改进计划，约束"上来就改" | ✅ 已用 |
| 5 | grill-me / grilling | 社区 | 对"示例 backlog 怎么补"做需求拷问（见演示） | ✅ 已演示 |
| 6 | gh-fix-ci | 官方 | 前置检查：项目无 CI 配置、gh 未登录 | ⛔ 条件不满足 |
| 7 | gh-address-comments | 官方 | 前置检查：无开放 PR、gh 未登录 | ⛔ 条件不满足 |
| 8 | tdd | 社区 | 把审计发现固化为回归检查脚本 | ✅ 已用 |
| 9 | diagnosing-bugs | 社区 | 系统化定位 2 个示例脚本语法 bug + 引用缺口 | ✅ 已用 |
| 10 | stop-slop | 社区 | 清洗两处典型 AI 腔总结句 | ✅ 已用 |
| 11 | sentry | 官方 | 前置检查：无 sentry CLI/项目，教程仓库无线上服务 | ⛔ 不适用 |
| 12 | book-to-skill | 社区 | 输入对象为书籍 PDF，本项目暂无输入 | ⚪ 待用 |
| 13 | playwright | 官方 | npx 已就绪；项目无 Web 页面可驱动 | ⚪ 待用 |
| 14 | cangjie-skill | 社区 | 输入对象为书/长视频，适合蒸馏教程素材 | ⚪ 待用 |
| 15 | last30days | 社区 | 可用于章节内容时效更新，需社交平台 API key | ⚪ 待用 |

判定原则：技能的 SKILL.md 自己要求先做前置检查（如 gh-fix-ci 要求先 `gh auth status`）。条件不满足时如实记录，不硬跑。

---

## 本次实际产生的改动

### 改动清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `scripts/check_examples.py` | 新增 | 示例健康检查：编译全部 .py + 校验 docs 引用路径，可复跑 |
| `examples/5-1-product/requirements_assessment.py` | 修复 | 第 267 行变量名笔误导致 SyntaxError |
| `examples/5-2-product/dialog_patterns.py` | 修复 | 第 243/245 行字符串内嵌 ASCII 双引号导致 SyntaxError |
| `examples/1-3-dl/activation_functions.py` | 新增 | 从 keypoints 文档抽取落地（文档早有代码、文件缺失） |
| `examples/1-3-dl/optimizer_comparison.py` | 新增 | 同上 |
| `examples/1-3-dl/training_template.py` | 新增 | 同上 |
| `examples/1-3-dl/README.md` | 更新 | 补登 3 个新示例 |
| `docs/chapter-3/README.md` | 改写 | stop-slop 清洗 KV 缓存总结句 |
| `docs/chapter-3/3-2-compression.md` | 改写 | stop-slop 清洗"总的来说"开场 |

---

### 案例 1：diagnosing-bugs —— 系统化修 bug，不靠猜

**纪律**：先收集证据、定位根因，再动手；禁止"猜一个原因改一版试试"。

**证据收集**（全量扫描，而非逐个点开）：

```bash
# 1. 编译全部示例，找出语法错误
find examples -name '*.py' -print0 | xargs -0 -I{} sh -c \
  'python3 -m py_compile "{}" 2>/dev/null || echo "FAIL: {}"'
# 结果：2/26 个文件失败

# 2. 校验文档引用的示例文件是否存在
grep -rhoE 'examples/[0-9A-Za-z_./-]+\.(py|sh|json)' docs --include='*.md' \
  | sort -u | while read p; do [ -e "$p" ] || echo "MISS $p"; done
# 结果：137 个引用中 120 个文件缺失（系统性问题，不是个别笔误）
```

**根因定位**：

| 现象 | 根因 | 修复 |
|------|------|------|
| `requirements_assessment.py` 语法错误 | 第 267 行 `unit econ = ...`，下划线被写成空格（AI 生成笔误） | 改为 `unit_econ` |
| `dialog_patterns.py` 语法错误 | 中文文案中嵌套了未转义的 ASCII `"`：`请回复"确认"继续` | 内层改用中文书名号 `「确认」` |
| 120 个示例文件缺失 | 文档以 `# examples/xxx.py` 注释内嵌代码，但文件从未抽取落地，属内容待建 backlog | 见案例 3，先建检查脚本防新增，再分批补 |

**修复后验证**（修完必须跑，不能只靠编译）：

```bash
python3 examples/5-1-product/requirements_assessment.py   # 输出完整 ROI/单位经济报告
python3 examples/5-2-product/dialog_patterns.py          # 45 行对话流程输出
# 全量复扫：26 个 .py 全部编译通过
```

### 案例 2：tdd —— 把发现变成回归测试

bug 会修完再犯。把审计逻辑沉淀为可复跑的检查脚本 `scripts/check_examples.py`：

- 硬检查：`examples/` 下所有 .py 必须通过编译，否则退出码 1
- 引用检查：`docs/` 中引用的 `examples/...` 路径必须存在；历史欠账默认只报告不拦截，`--strict` 模式下判失败
- 纯标准库，无需安装依赖

```bash
python3 scripts/check_examples.py            # 日常检查
python3 scripts/check_examples.py --strict   # CI 中使用：缺失引用也算失败
```

本次运行结果：26 个 Python 示例全部编译通过；缺失引用 117 个（修复前 120，补齐 1.3 节 3 个后下降），按章节分布打印，作为后续补示例的工单。

### 案例 3：补齐 1.3 节缺失示例

`docs/chapter-1/1-3-dl-keypoints.md` 内嵌了三段完整代码并标注了文件名，但文件不存在。从文档代码块原样抽取（脚本化，不手工誊抄），落地后逐个验证：

```bash
# activation_functions.py / optimizer_comparison.py 无头模式跑通
MPLBACKEND=Agg python3 examples/1-3-dl/activation_functions.py
MPLBACKEND=Agg python3 examples/1-3-dl/optimizer_comparison.py
# training_template.py 含完整训练流程，语法校验通过，按需手动运行
```

### 案例 4：stop-slop —— 清洗 AI 腔

规则：删开场废话、去"巧妙地/不可或缺"这类引语腔、说具体事实、用主动语态。

**改写 1**：`docs/chapter-3/README.md` KV 缓存小结

- 原文：
  > 总的来说，KV缓存是让大模型从"理论可行"走向"实际可用"的关键技术。它巧妙地平衡了计算与内存，是现代大模型部署中不可或缺的一环。
- 问题："总的来说"开场废话；"巧妙地平衡""不可或缺的一环"是空泛引语；没有一个具体事实。
- 改后：
  > 没有 KV 缓存，自回归生成每生成一个 token 都要重算全部历史 token 的注意力，开销随上下文长度二次增长。缓存 K、V 后，每步只算新位置，长对话和长文档功能在工程上才跑得动。

**改写 2**：`docs/chapter-3/3-2-compression.md` 决策路径开场

- 原文：`总的来说，可以根据你的具体情况，按以下路径进行决策：`
- 改后：`按你的场景选择路径：`

扫描方式：`grep -rhoE '值得注意的是|总的来说|综上所述|旨在|赋能|...' docs`。本项目文档 AI 腔密度不高，上述两处是最典型的残留；未做全仓大改，避免破坏作者原有表述。

### 案例 5：create-plan —— 先计划后动手

按 create-plan 的只读流程先扫描仓库（README、docs 结构、examples、CI 配置、git 状态），识别约束（无 CI、gh 未登录、纯 Markdown+Python 教程仓库），再按"发现 → 修改 → 验证"顺序执行，没有先写一行代码。本案例的执行顺序即计划产物。

### 案例 6：grill-me / grilling —— 需求拷问演示

grilling 技能要求把决策展开成"问题树"，每轮一次性问清当前边界问题并给出推荐答案。针对本项目"117 个缺失示例怎么补"的真实决策，第一轮拷问如下：

> ❓ **Q1 - 补示例的目标是什么？** 追求"文档引用全部存在"的数字好看，还是让学习者能跑通代码？
> ➡️ 推荐：以后者为准。能跑的 10 个好过抄出来的 100 个。
>
> ❓ **Q2 - 一次补多少？** 全仓 117 个一次性补完，还是按学习路径章节推进？
> ➡️ 推荐：按 AGENTS.md 的阶段清单推进（1-x → 2-x → 3-x），每章补完用 `check_examples.py` 验证。
>
> ❓ **Q3 - 代码从哪来？** 文档里已内嵌的直接抽取；文档没有的怎么办？
> ➡️ 推荐：内嵌代码用脚本抽取保证一致；没有的按章节新写，每个文件必须实际运行通过才算完成。
>
> ❓ **Q4 - 怎么防止再次腐烂？**
> ➡️ 推荐：把 `scripts/check_examples.py --strict` 挂到未来的 CI（配合 gh-fix-ci 技能）。

交互式使用方式：直接说"用 grill-me 拷问我这个需求"。

### 案例 7：skill-installer —— 安装全过程

```bash
# 官方精选（openai/skills 的 .curated 区）
python3 ~/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo openai/skills \
  --path skills/.curated/gh-fix-ci skills/.curated/gh-address-comments \
         skills/.curated/sentry skills/.curated/playwright \
  --dest ~/.agents/skills

# 社区技能（注意默认分支：create-plan / book-to-skill 在 master 分支，需 --ref master）
python3 .../install-skill-from-github.py \
  --repo composio-community/awesome-codex-skills --path create-plan \
  --ref master --dest ~/.agents/skills
```

踩坑记录：安装器默认拉 `main` 分支，默认分支为 `master` 的仓库会报临时目录冲突的 git 错误，加 `--ref master` 即可；`grill-me` 只是入口别名，实际逻辑在配套的 `grilling` 技能中，需一并安装。

---

## 技能逐一带过

### ⛔ 条件不满足 / 不适用（附触发条件）

| 技能 | 当前判定 | 条件满足后怎么用 |
|------|----------|------------------|
| gh-fix-ci | 仓库无 `.github/workflows/`，`gh auth status` 显示未登录 | 先 `gh auth login`（需 repo/workflow scope）；项目加 CI 后，PR 检查挂了直接说"修一下失败的 CI" |
| gh-address-comments | 当前分支无开放 PR | 推送分支开 PR 后，说"处理这个 PR 上的 review 意见" |
| sentry | 未安装 sentry CLI、无 SENTRY_AUTH_TOKEN；教程仓库无线上服务 | 第 4.6 节后端服务上线并接入 Sentry 后，说"看看最近 24h 的生产报错" |
| playwright | `npx` 已就绪（v11.7），但项目是 Markdown+Python，无 Web 页面 | 开发 4.5 节前端 AI 应用时，用它真机验证页面交互；也可用于截图检查文档渲染效果 |
| last30days | 需要社交平台 API key（ScrapeCreators/OpenAI 等可选注入） | 教程要求"紧跟技术动态"：说"调研最近 30 天 RAG 框架的社区讨论"，把结果补充到 4.3 节延伸阅读 |
| book-to-skill | 输入是书籍 PDF/EPUB，本次无输入 | 拿到教材 PDF 后说"把这本书做成 skill"，生成的技能放 `~/.agents/skills/` |
| cangjie-skill | 同上，且更适合蒸馏方法论类长内容 | 喂给它技术书/长视频文字稿，产出原子化能力卡技能包 |
| skill-creator | 本次没有需要新建的技能 | 当检查脚本需要在多个仓库复用时，可把 `scripts/check_examples.py` 包装成正式 skill |
| plugin-creator | 本次不需要分发插件 | 团队内部分发本项目技能集时打包 |

### 复现命令清单

```bash
# 1. 健康检查（回归测试）
python3 scripts/check_examples.py
python3 scripts/check_examples.py --strict   # CI 模式

# 2. 本次修复的两个示例
python3 examples/5-1-product/requirements_assessment.py
python3 examples/5-2-product/dialog_patterns.py

# 3. 新补的 1.3 节示例
MPLBACKEND=Agg python3 examples/1-3-dl/activation_functions.py
MPLBACKEND=Agg python3 examples/1-3-dl/optimizer_comparison.py

# 4. 查看已安装技能
ls ~/.agents/skills/
```

---

## 遗留 backlog 与后续触发场景

**Backlog**：`scripts/check_examples.py` 报告的 117 个"文档引用但缺失"的示例文件，按章节分布（2-2-pretraining 11 个、4-2-prompt 8 个、2-4/3-1/4-4 各 6-8 个居多）。这是教程内容建设的工单，不是 bug；建议按 AGENTS.md 的阶段顺序分批补齐，补完一章跑一次检查脚本。

**顺手发现但未改**：根目录 `README.md` 宣称"文档 54+、代码示例 65+"，实际为 62 个 Markdown、26 个 .py 示例；"内容完成度 100%"与 117 个缺失示例并存。数字口径由维护者决定如何更新，本次不擅自改写对外状态标识。

---

[← 返回番外篇目录](README.md)
DOCEF
echo "written: $(wc -l < /Users/weike/Documents/GitHub/AGI8926/AIAbilityGraph/docs/special-topics/special-05-codex-skills-practice.md) lines"