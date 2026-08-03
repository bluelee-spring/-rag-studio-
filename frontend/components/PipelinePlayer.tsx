"use client";

import { useEffect, useState } from "react";

import type { TraceStage } from "@/lib/types";
import { StageRenderer } from "./StageRenderer";

export function PipelinePlayer({
  stages,
  onStageChange,
}: {
  stages: TraceStage[];
  onStageChange?: (index: number) => void;
}) {
  const [activeIndex, setActiveIndex] = useState(0);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    setActiveIndex(0);
    setPlaying(true);
    onStageChange?.(0);
  }, [stages, onStageChange]);

  useEffect(() => {
    if (!playing || activeIndex >= stages.length - 1) return;
    const stageDuration: Record<string, number> = {
      "tfidf-build": 4_800,
      "cosine-workbench": 6_200,
      "bm25-document": 5_600,
      "bm25-accumulator": 5_000,
      "vector-index": 4_800,
      "dense-similarity": 5_200,
      "entity-space": 5_400,
      "graph-pattern": 5_400,
      "graph-traversal": 7_600,
      "graph-aggregate": 5_800,
      "triple-pattern": 5_400,
      "rdf-filter": 5_200,
      "row-filter": 5_200,
      "key-join": 5_400,
      "group-aggregate": 5_400,
    };
    const duration = stageDuration[stages[activeIndex]?.kind] ?? 3_600;
    const timer = window.setTimeout(() => {
      setActiveIndex((current) => {
        const next = Math.min(current + 1, stages.length - 1);
        onStageChange?.(next);
        if (next === stages.length - 1) setPlaying(false);
        return next;
      });
    }, duration);
    return () => window.clearTimeout(timer);
  }, [activeIndex, onStageChange, playing, stages]);

  if (!stages.length) return null;

  const stage = stages[activeIndex];

  function selectStage(index: number) {
    setActiveIndex(index);
    setPlaying(false);
    onStageChange?.(index);
  }

  function move(offset: number) {
    const next = Math.max(
      0,
      Math.min(stages.length - 1, activeIndex + offset),
    );
    selectStage(next);
  }

  return (
    <section className="pipeline-player" aria-label="RAG执行过程播放器">
      <header className="player-header">
        <div>
          <span className="player-kicker">RAG PROCESS</span>
          <strong>{stage.title}</strong>
        </div>
        <div className="player-controls">
          <button
            type="button"
            onClick={() => move(-1)}
            disabled={activeIndex === 0}
            aria-label="上一步"
          >
            ←
          </button>
          <button
            type="button"
            className="play-toggle"
            onClick={() => {
              if (activeIndex === stages.length - 1) {
                setActiveIndex(0);
                onStageChange?.(0);
              }
              setPlaying((current) => !current);
            }}
          >
            {playing ? "暂停" : activeIndex === stages.length - 1 ? "重播" : "播放"}
          </button>
          <button
            type="button"
            onClick={() => move(1)}
            disabled={activeIndex === stages.length - 1}
            aria-label="下一步"
          >
            →
          </button>
        </div>
      </header>

      <div className="stage-scrubber">
        {stages.map((item, index) => (
          <button
            type="button"
            key={item.id}
            className={
              index === activeIndex
                ? "scrubber-step active"
                : index < activeIndex
                  ? "scrubber-step complete"
                  : "scrubber-step"
            }
            onClick={() => selectStage(index)}
          >
            <i />
            <span>{item.title}</span>
          </button>
        ))}
      </div>

      <div className="visual-stage" key={stage.id}>
        <StageRenderer stage={stage} />
      </div>

      <footer className="player-footer">
        <span>
          {String(activeIndex + 1).padStart(2, "0")} /{" "}
          {String(stages.length).padStart(2, "0")}
        </span>
        <p>{stage.description || "观察数据如何进入下一阶段。"}</p>
        <small>
          {stage.status === "fallback"
            ? "本地教学执行"
            : stage.duration_ms > 0
              ? `${stage.duration_ms} ms`
              : "已完成"}
        </small>
      </footer>
    </section>
  );
}
