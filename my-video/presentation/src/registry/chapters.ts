import type { ChapterDef } from "./types";
import Coldopen from "../chapters/01-coldopen/Coldopen";
import { narrations as coldopenNarrations } from "../chapters/01-coldopen/narrations";
import ContentMap from "../chapters/02-content-map/ContentMap";
import { narrations as contentMapNarrations } from "../chapters/02-content-map/narrations";
import Highlights from "../chapters/03-highlights/Highlights";
import { narrations as highlightsNarrations } from "../chapters/03-highlights/narrations";
import LearningPath from "../chapters/04-learning-path/LearningPath";
import { narrations as learningPathNarrations } from "../chapters/04-learning-path/narrations";
import Action from "../chapters/05-action/Action";
import { narrations as actionNarrations } from "../chapters/05-action/narrations";

/**
 * Order = order of presentation.
 *
 * Each chapter MUST provide a `narrations: Narration[]` array. Its length
 * is the chapter's step count — there is no `totalSteps` to maintain
 * separately. This guarantees the audio synthesis pipeline, the runtime
 * stepper, and the chapter `.tsx` switch on `step` cannot drift apart.
 *
 * NOTE: chapter folders are self-checked in isolation during parallel
 * development (each with its own CSS prefix / folder). Only the order
 * and per-chapter narrations length need to be right here.
 *
 * Visual styling (color, fonts) comes entirely from the active theme —
 * chapters never hard-code palette / font names. See THEMES.md.
 */
export const CHAPTERS: ChapterDef[] = [
  {
    id: "coldopen",
    title: "开场钩子",
    narrations: coldopenNarrations,
    Component: Coldopen,
  },
  {
    id: "content-map",
    title: "七章内容地图",
    narrations: contentMapNarrations,
    Component: ContentMap,
  },
  {
    id: "highlights",
    title: "分量与四大亮点",
    narrations: highlightsNarrations,
    Component: Highlights,
  },
  {
    id: "learning-path",
    title: "三条学习路径",
    narrations: learningPathNarrations,
    Component: LearningPath,
  },
  {
    id: "action",
    title: "行动号召",
    narrations: actionNarrations,
    Component: Action,
  },
];
