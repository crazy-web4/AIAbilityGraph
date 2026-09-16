import type { ChapterStepProps } from "../../registry/types";
import "./LearningPath.css";

/**
 * 第 4 章 · learning-path —— 三条学习路径
 *
 * 4 步：
 *   0  三叉图：从「第 1 章」起点分岔出三条路径线（自绘）+ hero「三条学习路径」
 *   1  初学者：8–12 周，1→4→2→5（时间线节点依次点亮 + 连线自绘）
 *   2  工程师：10–14 周，1→2→3→4→6→7（路径更长、更密）
 *   3  产品经理：4–6 周，1→4→5→7（精简路径）
 *
 * 视觉演示：step0 = SVG 三岔自绘线；step1-3 = 横向章节节点时间线，节点按序点亮
 * + 连线自绘，其它两条路径灰化为上下文。全走 CSS transition / keyframes +
 * SVG stroke-dashoffset，无 setTimeout/setInterval。颜色字体全走主题 token。
 */

type RoleKey = "beginner" | "engineer" | "pm";

interface Path {
  key: RoleKey;
  name: string;
  en: string;
  weeks: string; // "8–12"
  weeksFull: string; // "8–12 周"
  order: number[]; // 章节访问顺序，以第 1 章为起点
  days: Record<number, number>;
  tag: string;
}

const PATHS: Path[] = [
  {
    key: "beginner",
    name: "初学者",
    en: "Beginner",
    weeks: "8–12",
    weeksFull: "8–12 周",
    order: [1, 4, 2, 5],
    days: { 1: 2, 4: 3, 2: 3, 5: 2 },
    tag: "零基础 · 平推进阶",
  },
  {
    key: "engineer",
    name: "工程师",
    en: "Engineer",
    weeks: "10–14",
    weeksFull: "10–14 周",
    order: [1, 2, 3, 4, 6, 7],
    days: { 1: 2, 2: 3, 3: 3, 4: 3, 6: 2, 7: 1 },
    tag: "前四章 + 6、7 章",
  },
  {
    key: "pm",
    name: "产品经理",
    en: "Product",
    weeks: "4–6",
    weeksFull: "4–6 周",
    order: [1, 4, 5, 7],
    days: { 1: 1, 4: 2, 5: 2, 7: 1 },
    tag: "看懂技术 · 想明白业务",
  },
];

const CH_NAME: Record<number, string> = {
  1: "基础理论",
  2: "核心技术",
  3: "工程化",
  4: "应用开发",
  5: "产品·业务",
  6: "算法工程",
  7: "资源·成本",
};

/* —— 共享几何 —— */
const SX = 150;
const SY = 285; // 共享起点（第 1 章）的 SVG 坐标
const LANE_Y = [110, 285, 460]; // 三条路径线的高度
const X0 = 430;
const XMAX = 1620;
const VIEW_W = 1780;
const VIEW_H = 560;

/* 该路径除第 1 章外的节点横坐标（在车道内等距排布） */
function laneXs(count: number): number[] {
  const xs: number[] = [];
  for (let i = 0; i < count; i++) {
    xs.push(count <= 1 ? X0 : X0 + Math.round((i * (XMAX - X0)) / (count - 1)));
  }
  return xs;
}

/* 从共享起点「分岔」到某车道、再沿着车道横穿的 SVG path */
function lanePath(y: number, xs: number[]): string {
  const first = xs[0];
  let d = `M${SX} ${SY} C ${SX + 150} ${SY}, ${first - 90} ${y}, ${first} ${y}`;
  for (let i = 1; i < xs.length; i++) d += ` L ${xs[i]} ${y}`;
  return d;
}

/* 每条路径的动画节奏（用时越短、越"精简"） */
interface Rhythm {
  lineMs: number;
  nodeStart: number;
  nodeGap: number;
}
const RHYTHM: Rhythm[] = [
  { lineMs: 1450, nodeStart: 500, nodeGap: 620 }, // 初学者
  { lineMs: 1750, nodeStart: 460, nodeGap: 470 }, // 工程师（长，细密）
  { lineMs: 1250, nodeStart: 560, nodeGap: 720 }, // 产品经理（短，从容）
];

export default function LearningPath({ step }: ChapterStepProps) {
  // steps 1-3 -> 0,1,2 ; step 0 -> -1
  const activeIdx = step - 1;
  const active = activeIdx >= 0 ? PATHS[activeIdx] : null;

  return (
    <div className="lp-scene">
      {/* ══ step 0 —— 三岔图引入 ══ */}
      {step === 0 && (
        <div className="lp-s0">
          <div className="lp-s0-head">
            <div className="lp-kicker">按身份 · 各配一条主线</div>
            <h1 className="lp-s0-title">
              <span className="hero-num lp-digits">三条</span>
              学习路径
            </h1>
            <p className="lp-s0-sub">同一张能力图谱，按你的来路，配一条线。</p>
          </div>

          <svg className="lp-s0-svg" viewBox="0 0 1780 430">
            {/* 共享起点：第 1 章 */}
            <g className="lp-broot">
              <circle cx={110} cy={215} r={30} className="lp-broot-ring" />
              <text x={110} y={222} textAnchor="middle" className="lp-broot-num">
                1
              </text>
              <text x={110} y={272} textAnchor="middle" className="lp-broot-tag">
                第 1 章 · 起点
              </text>
            </g>

            {/* 三条分岔线 */}
            {PATHS.map((p, i) => {
              const y = [100, 215, 330][i];
              const d = `M110 215 Q 540 215, 760 ${y} H 1660`;
              return (
                <g key={p.key} className="lp-branch" style={{ "--bi": i } as React.CSSProperties}>
                  <path className="lp-branch-line" d={d} />
                  <g className="lp-branch-end">
                    <text x={1660} y={y - 70} textAnchor="middle" className="lp-bname">
                      {p.name}
                    </text>
                    <text x={1660} y={y - 34} textAnchor="middle" className="lp-ben">
                      {p.en}
                    </text>
                    <text x={1660} y={y + 6} textAnchor="middle" className="lp-bweeks">
                      {p.weeksFull}
                    </text>
                  </g>
                </g>
              );
            })}
          </svg>
        </div>
      )}

      {/* ══ step 1-3 —— 逐条揭示 ══ */}
      {step >= 1 && active && (
        <div className="lp-s">
          <div className="lp-top">
            <div className="lp-chips">
              {PATHS.map((p, i) => (
                <span key={p.key} className={`lp-chip${i === activeIdx ? " on" : ""}`}>
                  {p.name}
                </span>
              ))}
            </div>

            <div className="lp-hero" key={active.key}>
              <div className="lp-role">
                <span className="lp-role-name">{active.name}</span>
                <span className="lp-role-tag">{active.tag}</span>
              </div>
              <div className="lp-weeks">
                <span className="hero-num lp-weeks-num">{active.weeks}</span>
                <span className="lp-weeks-unit">周</span>
              </div>
            </div>
          </div>

          <svg className="lp-lanes" viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}>
            {/* 三条骨架车道（上下文） */}
            {PATHS.map((p, i) => {
              const xs = laneXs(p.order.length - 1);
              return (
                <g key={p.key} className="lp-skel-g">
                  <path className="lp-skel" d={lanePath(LANE_Y[i], xs)} />
                  {xs.map((x, k) => (
                    <circle key={k} className="lp-skel-dot" cx={x} cy={LANE_Y[i]} r={14} />
                  ))}
                </g>
              );
            })}

            {/* 共享起点 · 第 1 章 */}
            <g className="lp-start">
              <circle cx={SX} cy={SY} r={32} className="lp-start-ring" />
              <text x={SX} y={SY} textAnchor="middle" className="lp-start-num">
                1
              </text>
              <text x={SX} y={SY + 46} textAnchor="middle" className="lp-start-tag">
                第 1 章
              </text>
            </g>

            {/* 当前路径：连线自绘 + 节点按序点亮 */}
            <g key={active.key} className="lp-active-g">
              {(() => {
                const y = LANE_Y[activeIdx];
                const xs = laneXs(active.order.length - 1);
                const nonStart = active.order.slice(1); // 第 1 章之后的章节
                const rh = RHYTHM[activeIdx];
                return (
                  <>
                    <path
                      className="lp-way"
                      d={lanePath(y, xs)}
                      style={{ animationDuration: `${rh.lineMs}ms` } as React.CSSProperties}
                    />
                    {nonStart.map((ch, k) => (
                      <g
                        key={ch}
                        className="lp-node-g"
                        style={
                          {
                            "--ld": `${rh.nodeStart + k * rh.nodeGap}ms`,
                          } as React.CSSProperties
                        }
                      >
                        <circle className="lp-node" cx={xs[k]} cy={y} r={20} />
                        <text className="lp-nnum" x={xs[k]} y={y + 7} textAnchor="middle">
                          {ch}
                        </text>
                        <text className="lp-nlabel" x={xs[k]} y={y - 46} textAnchor="middle">
                          第{ch}章 · {CH_NAME[ch]}
                        </text>
                        <text className="lp-week" x={xs[k]} y={y + 52} textAnchor="middle">
                          {active.days[ch]} 周
                        </text>
                      </g>
                    ))}
                  </>
                );
              })()}
            </g>
          </svg>
        </div>
      )}
    </div>
  );
}
