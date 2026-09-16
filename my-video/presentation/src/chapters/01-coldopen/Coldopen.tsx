import type { ChapterStepProps } from "../../registry/types";
import "./Coldopen.css";

/**
 * 第 1 章 · coldopen —— 开场钩子
 *
 * 3 步：
 *   0  钩子：上百个"知识点"散点混沌 + 疑问 hero「路线怎么理清？」
 *   1  定位：同一批散点归位成网格（混沌→有序），点出「AI 能力图谱」
 *   2  主线：横向自绘路径 5 节点，收尾「七个章节 · 完整覆盖」
 *
 * 视觉演示走 CSS transform / SVG stroke-dashoffset，颜色字体全走主题 token。
 */

/* 用索引哈希生成确定性伪随机（StrictMode 双渲染不闪） */
function hash(n: number, salt: number): number {
  const x = Math.sin(n * 127.1 + salt * 311.7) * 43758.5453;
  return x - Math.floor(x);
}

const DOT_COUNT = 120;
const GRID_COLS = 16;
const GRID_ROWS = 7;
const GRID_GAP_X = 96;
const GRID_GAP_Y = 96;
const GRID_OX = 150; // 网格原点（左上）
const GRID_OY = 295; // 让网格大致居中于左侧/全幅

interface Dot {
  gx: number; // grid col
  gy: number; // grid row
  sx: number; // scatter x（相对网格的偏移）
  sy: number; // scatter y
  sz: number; // dot 直径
}

const DOTS: Dot[] = Array.from({ length: DOT_COUNT }, (_, i) => {
  const gx = i % GRID_COLS;
  const gy = Math.floor(i / GRID_COLS) % GRID_ROWS;
  // 散点偏移：径向散开 + 噪声
  const spread = 720;
  const sx = Math.round((hash(i, 1) - 0.5) * 2 * spread);
  const sy = Math.round((hash(i, 2) - 0.5) * 2 * spread * 0.6);
  const sz = 5 + Math.round(hash(i, 3) * 6);
  return { gx, gy, sx, sy, sz };
});

export default function Coldopen({ step }: ChapterStepProps) {
  return (
    <div className="cd-scene">
      {/* 散点层：跨 step0/step1 持续存在，从混沌归位成网格 */}
      {(step === 0 || step === 1) && (
        <div className={`cd-dotfield${step === 1 ? " is-grid" : " is-chaos"}`}>
          {DOTS.map((d, i) => (
            <span
              key={i}
              className="cd-dot"
              style={
                {
                  left: GRID_OX + d.gx * GRID_GAP_X,
                  top: GRID_OY + d.gy * GRID_GAP_Y,
                  width: d.sz,
                  height: d.sz,
                  /* step0: 散到混沌位；step1: 归位网格（translate→0） */
                  "--dx": `${d.sx}px`,
                  "--dy": `${d.sy}px`,
                  "--i": i,
                } as React.CSSProperties
              }
            />
          ))}
        </div>
      )}

      {/* step 0 —— 钩子 */}
      {step === 0 && (
        <div className="cd-s1">
          <div className="cd-s1-body">
            <div className="cd-kicker" style={{ marginBottom: 40 }}>
              AI 全栈 · 从数学基础到落地应用
            </div>
            <h1 className="cd-s1-q">
              学 AI，要踩
              <br />
              <span className="cd-s1-em">几百个知识点</span>
              <br />
              路线，怎么一次理清？
            </h1>
            <div className="cd-s1-sub">
              从最基础的数学，一路走到能上线的应用。
            </div>
          </div>
        </div>
      )}

      {/* step 1 —— 定位：点出「AI 能力图谱」 */}
      {step === 1 && (
        <div className="cd-s2">
          <div className="cd-s2-mid">
            <div className="cd-kicker cd-s2-label">AI Ability Graph</div>
            <div className="cd-s2-name">
              <span className="cd-en">AI</span> 能力图谱
            </div>
            <div className="cd-s2-desc">
              不是散装的笔记堆。把整个 <strong>AI 全栈能力</strong>，
              按一条清晰的主线，<strong>排成一张图</strong>。
            </div>
          </div>
        </div>
      )}

      {/* step 2 —— 主线：横向路径 5 节点 */}
      {step === 2 && (
        <div className="cd-s3">
          <svg className="cd-s3-path" viewBox="0 0 1640 260">
            {/* 底导线 */}
            <path
              className="cd-s3-line"
              d="M40 130 H1600"
            />
            {/* 自绘主线（dashoffset 由 .in 触发） */}
            <path
              className={`cd-s3-line-fill${step === 2 ? " in" : ""}`}
              d="M40 130 H1600"
            />
            {[
              { t: "基础理论", x: 40 },
              { t: "核心技术", x: 440 },
              { t: "工程化", x: 840 },
              { t: "应用开发", x: 1240 },
              { t: "产品·资源", x: 1600 },
            ].map((n, i) => (
              <g key={n.t}>
                <circle
                  className="cd-s3-node in"
                  cx={n.x}
                  cy={130}
                  r={18}
                  style={{ transitionDelay: `${700 + i * 620}ms` }}
                />
                <text
                  className="cd-s3-label in"
                  x={n.x}
                  y={196}
                  textAnchor="middle"
                  style={{ transitionDelay: `${840 + i * 620}ms` }}
                >
                  {n.t}
                </text>
              </g>
            ))}
          </svg>
          <div className="cd-s3-title">
            <span className="cd-digits">七</span> 个章节 · 完整覆盖
          </div>
        </div>
      )}
    </div>
  );
}
