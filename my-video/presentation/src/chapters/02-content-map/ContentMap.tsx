import type { ChapterStepProps } from "../../registry/types";
import "./ContentMap.css";

/**
 * 第 2 章 · content-map —— 七章内容地图
 *
 * 9 步：两段式旅程。
 *   硬核技术段（0-4）：持久 7 节点横向地图轴上，第 1-4 章逐个点亮。
 *     0   建立全图 + 自绘一条"前四章硬核 / 后三章外溢"的切分线
 *     1-4 第 1/2/3/4 章逐步点亮，每章术语单独呈现（动作各不相同）
 *   外溢段（5-8）：step5 回旋——自绘"技术外溢"流线引入 5-7；随后三章逐一呈现
 *     5   引入后三章 + 自绘外溢流线
 *     6-8 第 5/6/7 章逐一呈现 + step8 收尾扫过全图
 *
 * 视觉演示：轴节点填充 / region 切分线自绘（dashoffset）/ 外溢流线自绘 + 流动
 * 光点 / step8 全图扫描。颜色与字体全走 token，前缀 .cm-。
 */

/* 7 章在地图轴上的横向位置（SVG viewBox 0 0 1920 340） */
const NODE_X = [260, 493, 726, 960, 1193, 1426, 1660];
const NODE_Y = 170;
const NODE_NAME = [
  "基础理论",
  "核心技术",
  "工程化",
  "应用开发",
  "产品·业务",
  "算法工程",
  "资源·成本",
];

/** 每章术语（画面信息密度 > 口播稿） */
const TERMS: string[][] = [
  ["数学统计", "经典 ML", "深度学习", "安全伦理"],
  ["架构", "预训练·微调", "PEFT · LoRA", "多模态"],
  ["分布式训练", "压缩量化", "推理优化", "数据处理", "MLOps", "评估 · 可解释"],
  ["API 开发", "提示词", "RAG", "Agent", "前端", "后端"],
  ["需求分析", "产品设计", "场景落地"],
  ["编程语言 · 框架", "算法工程化", "测试验证", "工程化部署"],
  ["算力规划", "模型服务容量", "GPU 调度"],
];

/** 每章文档 / 代码数量（口播未提及 → 让画面信息更密） */
const COUNTS: string[] = [
  "4 文档 · 6 可运行示例",
  "4 文档",
  "6 文档",
  "6 文档",
  "3 文档 · 1 示例",
  "4 文档 · 3 示例",
  "3 文档 · 3 示例",
];

/** step -> 该步点亮的章节（-1 表示无） */
const STEP_CH = [0, 0, 1, 2, 3, -1, 4, 5, 6]; // 0-based chapter idx

/* step0 与 step1…4 都聚焦第 1 章，但动作完全不同：0 只切分全图，1 才点亮第 1 章 */

function nodeClass(n: number, step: number): string {
  const hard = n <= 4; // n is 1-based index here
  if (step === 0) return hard ? "is-hard" : "is-soft";
  if (hard) {
    if (step >= 5) return "is-done";
    if (n < step) return "is-done";
    if (n === step) return "is-active";
    return "is-pre";
  }
  // n in 5..7
  if (step <= 4) return "is-soft";
  if (step === 5) return "is-intro";
  if (n < step) return "is-done";
  if (n === step) return "is-active";
  return "is-intro";
}

const CHAPTER_TITLE = [
  "AI 基础理论",
  "大模型核心技术",
  "大模型工程化",
  "大模型应用开发",
  "AI 产品与业务",
  "算法工程能力",
  "AI 资源与成本",
];
const CHAPTER_TAG = [
  "理论与数学 · 全部 AI 的地基",
  "从架构到多模态 · 一层层往里挖",
  "规模训练到推理 · 工程侧该有的都有",
  "让能力变成产品 · 真正上手做产品的地方",
  "让技术外溢成业务 · 需求到落地",
  "通用工程能力 · 从写码到部署",
  "算力与容量 · 把资源花在刀刃上",
];

export default function ContentMap({ step }: ChapterStepProps) {
  const ch = STEP_CH[step]; // -1 for step0 / step5

  return (
    <div className="cm-scene">
      {/* ── 持久地图轴：7 章横轴，贯穿全部 9 步 ── */}
      <svg className="cm-axis" viewBox="0 0 1920 340" aria-hidden="true">
        {/* 横轴底导线 */}
        <line className="cm-rail" x1={NODE_X[0]} y1={NODE_Y} x2={NODE_X[6]} y2={NODE_Y} />

        {/* step0：自绘切分线 —— 前四章硬核 / 后三章外溢 */}
        <line
          className={`cm-divider${step === 0 ? " in" : ""}`}
          x1={1072}
          y1={30}
          x2={1072}
          y2={NODE_Y + 112}
        />

        {/* step5：自绘外溢流线（1-4 → 5-7）+ 流动光点 */}
        {step >= 5 && (
          <g>
            <path
              className={`cm-flow${step === 5 ? " in" : ""}`}
              d="M 958 120 Q 1070 32 1205 120"
              fill="none"
            />
            <path
              className={`cm-flow-head${step === 5 ? " in" : ""}`}
              d="M 1186 132 L 1220 120 L 1186 108 Z"
            />
            <circle className="cm-flow-spark" cx={0} cy={0} r="0" />
          </g>
        )}

        {/* 7 个章节节点 */}
        {NODE_X.map((x, i) => {
          const cls = nodeClass(i + 1, step);
          return (
            <g key={i} className={`cm-node ${cls}`} style={{ transitionDelay: "" }}>
              <circle
                className="cm-node-ring"
                cx={x}
                cy={NODE_Y}
                r={26}
                style={{ transitionDelay: `${(i + 1) * 90}ms` }}
              />
              <text className="cm-node-num" x={x} y={NODE_Y + 10} textAnchor="middle">
                {i + 1}
              </text>
              <text className="cm-node-name" x={x} y={NODE_Y + 58} textAnchor="middle">
                {NODE_NAME[i]}
              </text>
            </g>
          );
        })}

        {/* 分区标签 */}
        <text
          className={`cm-region cm-region-hard${step === 0 ? " in" : ""}`}
          x={600}
          y={290}
          textAnchor="middle"
        >
          硬核技术 · HARDCORE
        </text>
        <text
          className={`cm-region cm-region-soft${step === 0 || step === 5 ? " in" : ""}`}
          x={1430}
          y={290}
          textAnchor="middle"
        >
          技术外溢 · OUTFLOW
        </text>
      </svg>

      {/* ── 顶部叙事层 ── */}
      {/* step 0 —— 建图 + 切分两段 */}
      {step === 0 && (
        <div className="cm-s cm-s0">
          <div className="cm-kicker">AI 能力图谱 · 七章内容地图</div>
          <h1 className="cm-hero">
            硬核技术
            <br />
            集中在前<span className="cm-ch-num">四</span>章
          </h1>
          <div className="cm-sub">
            前四章打地基、挖技术、扛工程、做产品；后三章，让技术外溢成资源与业务。
          </div>
        </div>
      )}

      {/* steps 1-4 与 6-8 —— 逐章点亮 + 术语独立呈现 */}
      {ch >= 0 && (
        <div className="cm-s cm-s-ch">
          <div className="cm-kicker">第 {ch + 1} 章 · 七章之一</div>
          <h1 className="cm-hero">
            <span className="cm-hero-num cm-hero-num-1 in">{ch + 1}</span>
            {CHAPTER_TITLE[ch]}
          </h1>
          <div className="cm-tag">{CHAPTER_TAG[ch]}</div>
          <div className={`cm-chiprow cm-ct-${ch + 1}`}>
            {TERMS[ch].map((t, i) => (
              <span key={t} className="cm-chip" style={{ "--i": i } as React.CSSProperties}>
                {t}
              </span>
            ))}
          </div>
          <div className="cm-count">{COUNTS[ch]}</div>
        </div>
      )}

      {/* step 5 —— 回旋：技术外溢，流向后三章 */}
      {step === 5 && (
        <div className="cm-s cm-s5">
          <div className="cm-kicker">转折 · 前四章 → 后三章</div>
          <h1 className="cm-hero">
            技术，开始
            <span className="cm-ch-num">外溢</span>
          </h1>
          <div className="cm-sub">
            溢出到 <strong>产品</strong>、<strong>工程</strong> 与 <strong>资源</strong> —— 让一套
            能力图谱，落地成真正可运转的东西。
          </div>
        </div>
      )}

      {/* step 8 —— 收尾：全图扫描 */}
      {step === 8 && <div className="cm-sweep" />}
    </div>
  );
}
