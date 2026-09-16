import type { ChapterStepProps } from "../../registry/types";
import "./Highlights.css";

/**
 * 第 3 章 · highlights —— 分量与四大亮点
 *
 * 6 步（maxStep 5）：
 *   0  分量：hero 标语 + 四大数据「齿轮滚动式 count-up」（38+ / 15+ / 12万+ / 100%）
 *   1  骨架：点出「四个亮点」，左侧 4 槽空位（只放序号，槽位全空）
 *   2  亮点一「成体系」：亮起槽1 + 七章链条 SVG 自绘（基础→高级）
 *   3  亮点二「代码够多」：槽1 灰化 / 槽2 亮 + 代码行生长 demo
 *   4  亮点三「实战导向」：槽2 灰化 / 槽3 亮 + 节点流程箭头自绘
 *   5  亮点四「工程化」：槽3 灰化 / 槽4 亮 + 五级 pipeline 进度条
 *
 * 视觉演示全走 CSS transform / SVG stroke-dashoffset / keyframes，
 * 颜色字体全走主题 token（前缀 .hl-）。无 setTimeout/setInterval。
 */

/* ── 分量数字（step 0）—— 键盘滚动式 count-up ── */
const DIGITS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
const STATS: {
  digits: number[];
  unit: string;
  note: string;
}[] = [
  { digits: [3, 8], unit: "+", note: "个以上 · 文档" },
  { digits: [1, 5], unit: "+", note: "个 · 可运行代码示例" },
  { digits: [1, 2], unit: "万+", note: "字 · 总字数" },
  { digits: [1, 0, 0], unit: "％", note: "每章都是 · 100% 完成" },
];

/* ── 四大亮点（step 1-5）── */
const HIGHLIGHTS = [
  {
    short: "成体系",
    big: "成体系",
    sub: "七章从基础到高级，串成一条完整价值链",
    points: ["七章 · 基础到高级", "一条完整价值链", "不是散装的笔记堆"],
  },
  {
    short: "代码够多",
    big: "代码 · 每个知识点都能跑起来",
    sub: "数学 / 机器学习 / 深度学习，都有直接可运行的示例撑着",
    points: ["数学基础", "机器学习", "深度学习"],
    foot: "15+ 个可运行代码文件",
  },
  {
    short: "实战导向",
    big: "实战 · 每一条都有完整流程",
    sub: "API 开发、RAG 系统、Agent 架构、前后端应用",
    points: ["API 开发", "RAG 系统", "Agent 架构", "前后端应用"],
  },
  {
    short: "工程化",
    big: "工程化 · 全篇摊开讲",
    sub: "训练/推理/成本/资源/交付，都讲透了",
    points: ["分布式训练", "推理优化 · KV Cache 批处理", "成本优化", "多租户资源管理", "CI/CD"],
  },
];

/* 七章链条节点（亮点一 demo） */
const CHAIN = [
  "基础理论",
  "核心技术",
  "工程化",
  "应用开发",
  "产品与业务",
  "算法工程",
  "资源与成本",
];

/* 实战流程节点（亮点三 demo） */
const FLOW = ["API 开发", "RAG 系统", "Agent 架构", "前后端应用"];

/* 工程化五级 pipeline（亮点四 demo） */
const PLINE = ["分布式训练", "推理优化", "成本优化", "多租户", "CI/CD"];

export default function Highlights({ step }: ChapterStepProps) {
  /* 当前亮点下标：step2→0 ... step5→3；step1 为骨架（-1） */
  const cur = step - 2;

  return (
    <div className="hl-scene">
      {/* ════════════ STEP 0 · 分量 —— 数据 count-up ════════════ */}
      {step === 0 && (
        <div className="hl-s0">
          <div className="hl-kicker">分量 · Volume</div>
          <h2 className="hl-s0-head">
            整套做下来，<em>分量相当足</em>
          </h2>
          <div className="hl-stats">
            {STATS.map((s, k) => (
              <div key={k} className="hl-stat" style={{ "--k": k } as React.CSSProperties}>
                <div className="hl-od">
                  {s.digits.map((d, i) => (
                    <span
                      key={i}
                      className="hl-dcell"
                      style={{ "--t": d, "--d": `calc(var(--k) * 450ms + ${i} * 120ms)` } as React.CSSProperties}
                    >
                      <span className="hl-dstrip">
                        {DIGITS.map((v) => (
                          <span key={v} className="hl-drow">
                            {v}
                          </span>
                        ))}
                      </span>
                    </span>
                  ))}
                  <span className="hl-unit" style={{ "--d": `calc(var(--k) * 450ms + ${s.digits.length} * 120ms)` } as React.CSSProperties}>
                    {s.unit}
                  </span>
                </div>
                <div className="hl-note" style={{ "--d": `calc(var(--k) * 450ms + ${s.digits.length} * 140ms)` } as React.CSSProperties}>
                  {s.note}
                </div>
              </div>
            ))}
          </div>
          <div className="hl-s0-foot">七章 · 100% 完成</div>
        </div>
      )}

      {/* ════════════ STEP 1-5 · 四大亮点（骨架 → 逐个填槽） ════════════ */}
      {step >= 1 && (
        <div className="hl-s4">
          <div className="hl-s4-head">
            <div className="hl-kicker">四大亮点 · Four Highlights</div>
            <h2 className="hl-s4-title">四个，特别值得你关注的亮点</h2>
          </div>

          <div className="hl-s4body">
            {/* 左侧 · 4 槽列表（1 项 1 step 逐个揭示） */}
            <div className="hl-rail">
              {HIGHLIGHTS.map((h, i) => {
                let state = "next";
                if (i < cur) state = "done";
                else if (i === cur) state = "cur";
                return (
                  <div key={i} className={`hl-item is-${state}`}>
                    <span className="hl-badge">{String(i + 1).padStart(2, "0")}</span>
                    <span className="hl-item-label">{i <= cur ? h.short : " "}</span>
                  </div>
                );
              })}
            </div>

            {/* 右侧 · 当前亮点的细节 + 迷你演示 */}
            {cur >= 0 && (
              <div className="hl-detail" key={cur}>
                <div className="hl-detail-head">
                  <h3 className="hl-detail-title">{HIGHLIGHTS[cur].big}</h3>
                  <div className="hl-detail-sub">{HIGHLIGHTS[cur].sub}</div>
                </div>

                <div className="hl-pts">
                  {HIGHLIGHTS[cur].points.map((p, i) => (
                    <span key={i} className="hl-pt" style={{ "--i": i } as React.CSSProperties}>
                      {p}
                    </span>
                  ))}
                </div>

                {/* 亮点一：七章链条自绘 */}
                {cur === 0 && (
                  <div className="hl-demo hl-demo-chain">
                    <svg viewBox="0 0 1240 210" className="hl-chain">
                      <path
                        className="hl-chain-bg"
                        d={`M${chX(0)} 96 H${chX(6)}`}
                      />
                      <path
                        className="hl-chain-fill"
                        d={`M${chX(0)} 96 H${chX(6)}`}
                      />
                      {CHAIN.map((c, i) => (
                        <g key={c} style={{ "--i": i } as React.CSSProperties}>
                          <circle className="hl-chain-node" cx={chX(i)} cy={96} r={16} />
                          <text className="hl-chain-label" x={chX(i)} y={140} textAnchor="middle">
                            {c}
                          </text>
                        </g>
                      ))}
                    </svg>
                    <div className="hl-direction">
                      <span>基础</span>
                      <span className="hl-dir-arrow">→</span>
                      <span>高级</span>
                    </div>
                  </div>
                )}

                {/* 亮点二：代码行生长 */}
                {cur === 1 && (
                  <div className="hl-demo hl-demo-code">
                    <div className="hl-code-tags">
                      {HIGHLIGHTS[1].points.map((t, i) => (
                        <span key={t} className="hl-code-tag" style={{ "--i": i } as React.CSSProperties}>
                          {t}
                        </span>
                      ))}
                    </div>
                    <div className="hl-code-panel">
                      {[0, 1, 2, 3, 4].map((r) => (
                        <div key={r} className={`hl-code-line lv${(r % 3) + 1}`} style={{ "--i": r } as React.CSSProperties}>
                          <span className="hl-code-caret" />
                          <span className="hl-code-bar" />
                        </div>
                      ))}
                    </div>
                    <div className="hl-code-foot">15+ 个可运行代码文件</div>
                  </div>
                )}

                {/* 亮点三：流程节点箭头自绘 */}
                {cur === 2 && (
                  <div className="hl-demo hl-demo-flow">
                    <svg viewBox="0 0 1240 170" className="hl-flow">
                      {FLOW.map((f, i) => {
                        const x0 = 100 + i * 300;
                        const x1 = 100 + (i + 1) * 300;
                        return (
                          <g key={f} style={{ "--i": i } as React.CSSProperties}>
                            {i < FLOW.length - 1 && (
                              <g>
                                <path className="hl-flow-line" d={`M${x0 + 165} 92 H${x1 - 45}`} />
                                <path
                                  className="hl-flow-arrow"
                                  d={`M${x1 - 45} 92 l-15 8 M${x1 - 45} 92 l-15 -8`}
                                />
                              </g>
                            )}
                            <rect className="hl-flow-node" x={x0} y={63} width={165} height={58} rx={9} />
                            <text className="hl-flow-label" x={x0 + 82} y={98} textAnchor="middle">
                              {f}
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                  </div>
                )}

                {/* 亮点四：五级 pipeline 进度 */}
                {cur === 3 && (
                  <div className="hl-demo hl-demo-pl">
                    <div className="hl-pl-track">
                      <span className="hl-pl-bar" />
                      <span className="hl-pl-pulse" />
                    </div>
                    <div className="hl-pl-stages">
                      {PLINE.map((p, i) => (
                        <span key={p} className="hl-pl-stage" style={{ "--i": i } as React.CSSProperties}>
                          {p}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {cur >= 0 && (
            <div className="hl-slug">
              <span className="hl-slug-num">0{cur + 1}</span>
              <span className="hl-slug-txt">亮点</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* 链条节点横坐标（均匀分布） */
function chX(i: number): number {
  return 46 + i * 192;
}
