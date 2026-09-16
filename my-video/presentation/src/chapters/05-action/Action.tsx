import type { ChapterStepProps } from "../../registry/types";
import "./Action.css";

/**
 * 第 5 章 · action —— 行动号召收尾（视频 finale，干净利落地收）
 *
 * 3 步：
 *   0  从 README 开始 + MIT License：一个 README 文档节点，经一条
 *      自绘路径穿入"开源许可证大门"，揭示「自由学 · 自由用」。
 *   1  号召：抽象 Star 按钮 + 光标按压动效 + 水波纹扩散 + 星被点亮，
 *      配「点个 Star，持续更新的最大动力」。（不伪造 star 数量 / 头像）
 *   2  收尾大标语：从第 1 章开始，一起把 AI 学到底（字母错峰 + 底线自绘，
 *      与开场呼应，bookend）。
 *
 * 视觉演示全走 CSS keyframes / transition（无 setTimeout/setInterval），
 * 颜色字体全走主题 token，前缀 .ac-。
 */

export default function Action({ step }: ChapterStepProps) {
  if (step === 0) {
    return (
      <div className="ac-scene">
        {/* step 0 —— 从 README 开始 + MIT License 自由学自由用 */}
        <div className="ac-s0">
          <div className="ac-kicker ac-s0-kicker">ACTION · 从 README 开始</div>

          <svg
            className="ac-s0-svg"
            viewBox="0 0 1640 560"
            aria-hidden="true"
          >
            {/* 底导线：README → 许可证大门（静止虚线） */}
            <path
              className="ac-s0-conn-base"
              d="M300 300 C 740 30, 1000 420, 1282 360"
            />
            {/* 自绘主线：由 README 门口画出，穿入大门开口 */}
            <path
              className="ac-s0-conn-fill"
              d="M300 300 C 740 30, 1000 420, 1282 360"
            />

            {/* README 文档节点（入口） */}
            <g className="ac-s0-file">
              <rect x="152" y="150" width="148" height="210" rx="10" />
              <path d="M228 150 V238 H300" />
            </g>
            <text className="ac-s0-filet" x="226" y="520" textAnchor="middle">
              README 入口
            </text>

            {/* 开源许可证大门（拱形）+ 自绘 */}
            <path
              className="ac-s0-arch-base"
              d="M1156 470 L1156 262 Q1156 150 1295 150 Q1434 150 1434 262 L1434 470"
            />
            <path
              className="ac-s0-arch-fill"
              d="M1156 470 L1156 262 Q1156 150 1295 150 Q1434 150 1434 262 L1434 470"
            />
            <text className="ac-s0-gatein" x="1295" y="420" textAnchor="middle">
              自由
            </text>
          </svg>

          {/* 许可证徽记 */}
          <div className="ac-s0-lic">
            <div className="ac-s0-lic-title">MIT License</div>
            <div className="ac-s0-lic-sub">自由学 · 自由用</div>
            <div className="ac-s0-lic-detail">文档 CC BY-SA　·　代码 MIT</div>
          </div>
        </div>
      </div>
    );
  }

  if (step === 1) {
    return (
      <div className="ac-scene">
        {/* step 1 —— 号召：点 Star / 一起贡献 */}
        <div className="ac-s1">
          <div className="ac-kicker ac-s1-kicker">CALL TO ACTION · 一起贡献</div>

          <div className="ac-s1-stage">
            {/* 抽象光标箭头：飞入后按向 Star 按钮 */}
            <svg
              className="ac-s1-cursor"
              viewBox="0 0 44 44"
              aria-hidden="true"
            >
              <path
                d="M6 3 L38 30 L22 31 L17 42 L11 29 L3 25 Z"
                transform="rotate(6 22 22)"
              />
            </svg>

            {/* Star 按钮：按压 → 水波纹扩散 → 星点亮 */}
            <button
              type="button"
              data-no-advance
              tabIndex={-1}
              className="ac-s1-btn"
            >
              <svg
                className="ac-s1-star"
                viewBox="0 0 32 32"
                aria-hidden="true"
              >
                <path
                  d="M16 2.4 L19.6 12.2 L30 12.5 L21.2 19 L24.4 29 L16 23 L7.6 29 L10.8 19 L2 12.5 L12.4 12.2 Z"
                />
              </svg>
              <span className="ac-s1-btnlabel">Star</span>
            </button>

            {/* 波纹扩散层 */}
            <span className="ac-s1-ripple" aria-hidden="true" />
          </div>

          <div className="ac-s1-line">
            <span className="ac-s1-line-em">点个 Star</span>
            <span className="ac-s1-line-sub">持续更新的最大动力</span>
          </div>
        </div>
      </div>
    );
  }

  /* step 2 —— 收尾大标语（bookend：呼应开场"一起学"） */
  return (
    <div className="ac-scene">
      <div className="ac-s2">
        <div className="ac-kicker ac-s2-kicker">ENJOY · 一路学到底</div>

        <h2 className="ac-s2-hero">
          <span className="ac-s2-row">
            <span className="ac-s2-word">从</span>
            <span className="ac-s2-hl">第 1 章</span>
            <span className="ac-s2-word">开始</span>
          </span>
          <span className="ac-s2-row ac-s2-row2">
            <span className="ac-s2-word">一起把 AI 学到底</span>
          </span>
        </h2>

        <div className="ac-s2-rule" aria-hidden="true" />

        <div className="ac-s2-tail">—— AI 能力图谱 ——</div>
      </div>
    </div>
  );
}
