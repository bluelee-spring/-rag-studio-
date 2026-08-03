"use client";

import { useEffect, useMemo, useRef } from "react";

export type SpatialPoint = {
  id: string;
  label: string;
  type: string;
  x?: number;
  y?: number;
  z?: number;
  score?: number;
};

export type SpatialEdge = {
  source: string;
  target: string;
  relation: string;
};

type PositionedPoint = SpatialPoint & {
  x: number;
  y: number;
  z: number;
};

const TYPE_COLORS: Record<string, string> = {
  Query: "#65f4c3",
  Symptom: "#72d8ff",
  DiseaseCase: "#a9b7b3",
  Disease: "#ffbf70",
  Region: "#c69cff",
  Pesticide: "#f98da4",
  Document: "#f5e77d",
  Resource: "#9cb6ff",
  variable: "#72d8ff",
  iri: "#ffbf70",
  literal: "#c69cff",
};

function hashNumber(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return Math.abs(hash);
}

function graphPosition(
  point: SpatialPoint,
  typeIndex: number,
  typeCount: number,
): PositionedPoint {
  if (
    typeof point.x === "number" &&
    typeof point.y === "number" &&
    typeof point.z === "number"
  ) {
    return {
      ...point,
      x: point.x * 250,
      y: point.y * 170,
      z: point.z * 190,
    };
  }

  const seed = hashNumber(point.id);
  const angle =
    typeIndex * 2.399963229728653 +
    ((seed % 100) / 100) * 0.45;
  const radius = 48 + Math.sqrt(typeIndex + 1) * 23;
  const spread = Math.max(typeCount, 1);

  if (point.type === "DiseaseCase") {
    return {
      ...point,
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius * 0.66,
      z: ((typeIndex % 7) - 3) * 31,
    };
  }

  const cluster: Record<string, [number, number, number]> = {
    Query: [-320, -20, 30],
    Symptom: [-255, 95, 70],
    Region: [-255, -115, -70],
    Disease: [245, -70, 80],
    Pesticide: [305, 92, -40],
    Document: [365, 145, 70],
    Resource: [250, 120, -80],
    variable: [-50, -30, 20],
    iri: [230, 20, -40],
    literal: [80, 145, 60],
  };
  const base = cluster[point.type] ?? [190, 100, 0];
  const offset = (typeIndex - (spread - 1) / 2) * 56;
  return {
    ...point,
    x: base[0],
    y: base[1] + offset,
    z: base[2] + ((seed % 5) - 2) * 22,
  };
}

function withPositions(points: SpatialPoint[]): PositionedPoint[] {
  const counters = new Map<string, number>();
  const totals = new Map<string, number>();
  for (const point of points) {
    totals.set(point.type, (totals.get(point.type) ?? 0) + 1);
  }
  return points.map((point) => {
    const index = counters.get(point.type) ?? 0;
    counters.set(point.type, index + 1);
    return graphPosition(
      point,
      index,
      totals.get(point.type) ?? 1,
    );
  });
}

export function SpatialScene({
  points,
  edges = [],
  selectedId,
  activeIds = [],
  activeRelation,
  mode = "graph",
  ariaLabel,
}: {
  points: SpatialPoint[];
  edges?: SpatialEdge[];
  selectedId?: string;
  activeIds?: string[];
  activeRelation?: string;
  mode?: "graph" | "embedding" | "vector";
  ariaLabel: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const interactionRef = useRef({
    pointerX: 0,
    pointerY: 0,
    dragging: false,
    lastX: 0,
    lastY: 0,
    manualYaw: 0,
    manualPitch: 0,
  });
  const positioned = useMemo(() => withPositions(points), [points]);
  const activeSet = useMemo(() => new Set(activeIds), [activeIds]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    let frame = 0;
    let width = 0;
    let height = 0;
    let pixelRatio = 1;

    function resize() {
      if (!canvas || !context) return;
      const bounds = canvas.getBoundingClientRect();
      width = Math.max(320, bounds.width);
      height = Math.max(360, bounds.height);
      pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * pixelRatio);
      canvas.height = Math.round(height * pixelRatio);
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    }

    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    resize();

    function project(
      point: PositionedPoint,
      yaw: number,
      pitch: number,
    ) {
      const cosY = Math.cos(yaw);
      const sinY = Math.sin(yaw);
      const x1 = point.x * cosY - point.z * sinY;
      const z1 = point.x * sinY + point.z * cosY;
      const cosX = Math.cos(pitch);
      const sinX = Math.sin(pitch);
      const y1 = point.y * cosX - z1 * sinX;
      const z2 = point.y * sinX + z1 * cosX;
      const focal = 820;
      const scale = focal / (focal + z2 + 260);
      return {
        x: width / 2 + x1 * scale,
        y: height / 2 + y1 * scale,
        z: z2,
        scale,
      };
    }

    function drawGrid(yaw: number, pitch: number) {
      if (!context) return;
      context.save();
      context.lineWidth = 1;
      for (let row = -5; row <= 5; row += 1) {
        const start = project(
          {
            id: "",
            label: "",
            type: "",
            x: -430,
            y: 215,
            z: row * 70,
          },
          yaw,
          pitch,
        );
        const end = project(
          {
            id: "",
            label: "",
            type: "",
            x: 430,
            y: 215,
            z: row * 70,
          },
          yaw,
          pitch,
        );
        const gradient = context.createLinearGradient(
          start.x,
          start.y,
          end.x,
          end.y,
        );
        gradient.addColorStop(0, "rgba(84, 235, 186, 0)");
        gradient.addColorStop(0.5, "rgba(84, 235, 186, 0.11)");
        gradient.addColorStop(1, "rgba(84, 235, 186, 0)");
        context.strokeStyle = gradient;
        context.beginPath();
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
        context.stroke();
      }
      for (let column = -6; column <= 6; column += 1) {
        const start = project(
          {
            id: "",
            label: "",
            type: "",
            x: column * 70,
            y: 215,
            z: -360,
          },
          yaw,
          pitch,
        );
        const end = project(
          {
            id: "",
            label: "",
            type: "",
            x: column * 70,
            y: 215,
            z: 360,
          },
          yaw,
          pitch,
        );
        context.strokeStyle = "rgba(138, 168, 159, 0.08)";
        context.beginPath();
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
        context.stroke();
      }
      context.restore();
    }

    function draw(timestamp: number) {
      if (!canvas || !context) return;
      const interaction = interactionRef.current;
      const idleYaw =
        mode === "embedding"
          ? Math.sin(timestamp / 9000) * 0.18
          : Math.sin(timestamp / 12000) * 0.1;
      const yaw =
        idleYaw +
        interaction.manualYaw +
        interaction.pointerX * 0.16;
      const pitch =
        -0.13 +
        interaction.manualPitch +
        interaction.pointerY * 0.1;

      context.clearRect(0, 0, width, height);
      const backdrop = context.createRadialGradient(
        width * 0.52,
        height * 0.43,
        20,
        width * 0.52,
        height * 0.43,
        Math.max(width, height) * 0.7,
      );
      backdrop.addColorStop(0, "rgba(23, 70, 58, 0.42)");
      backdrop.addColorStop(0.46, "rgba(7, 24, 20, 0.93)");
      backdrop.addColorStop(1, "rgba(3, 10, 9, 1)");
      context.fillStyle = backdrop;
      context.fillRect(0, 0, width, height);
      drawGrid(yaw, pitch);

      const projected = new Map<
        string,
        ReturnType<typeof project> & { point: PositionedPoint }
      >();
      for (const point of positioned) {
        projected.set(point.id, {
          ...project(point, yaw, pitch),
          point,
        });
      }

      const sortedEdges = edges
        .map((edge, index) => {
          const source = projected.get(edge.source);
          const target = projected.get(edge.target);
          return source && target
            ? {
                edge,
                source,
                target,
                depth: (source.z + target.z) / 2,
                index,
              }
            : null;
        })
        .filter((item): item is NonNullable<typeof item> => Boolean(item))
        .sort((left, right) => left.depth - right.depth);

      for (const item of sortedEdges) {
        const relationActive =
          !activeRelation ||
          item.edge.relation === activeRelation;
        const idsActive =
          activeSet.size === 0 ||
          activeSet.has(item.edge.source) ||
          activeSet.has(item.edge.target);
        const active = relationActive && idsActive;
        const gradient = context.createLinearGradient(
          item.source.x,
          item.source.y,
          item.target.x,
          item.target.y,
        );
        gradient.addColorStop(
          0,
          active
            ? "rgba(91, 246, 194, 0.74)"
            : "rgba(127, 158, 149, 0.16)",
        );
        gradient.addColorStop(
          1,
          active
            ? "rgba(111, 201, 255, 0.55)"
            : "rgba(127, 158, 149, 0.08)",
        );
        context.strokeStyle = gradient;
        context.lineWidth = active ? 1.7 : 0.8;
        context.beginPath();
        context.moveTo(item.source.x, item.source.y);
        context.lineTo(item.target.x, item.target.y);
        context.stroke();

        if (active && item.index < 36) {
          const progress =
            (timestamp / 1500 + item.index * 0.173) % 1;
          const x =
            item.source.x +
            (item.target.x - item.source.x) * progress;
          const y =
            item.source.y +
            (item.target.y - item.source.y) * progress;
          context.save();
          context.shadowBlur = 16;
          context.shadowColor = "#65f4c3";
          context.fillStyle = "#c7ffed";
          context.beginPath();
          context.arc(x, y, 2.4, 0, Math.PI * 2);
          context.fill();
          context.restore();
        }
      }

      const sortedPoints = [...projected.values()].sort(
        (left, right) => left.z - right.z,
      );
      for (const item of sortedPoints) {
        const point = item.point;
        const isSelected = point.id === selectedId;
        const isActive =
          activeSet.size === 0 || activeSet.has(point.id);
        const baseColor =
          TYPE_COLORS[point.type] ?? "#89b5aa";
        const radius =
          (point.type === "DiseaseCase" ? 4.2 : 8.2) *
          item.scale *
          (isSelected ? 1.5 : 1);
        const alpha = isActive ? 1 : 0.24;

        context.save();
        context.globalAlpha = alpha;
        context.shadowBlur = isSelected ? 30 : isActive ? 14 : 4;
        context.shadowColor = baseColor;
        const halo = context.createRadialGradient(
          item.x - radius * 0.35,
          item.y - radius * 0.35,
          0,
          item.x,
          item.y,
          radius * 1.8,
        );
        halo.addColorStop(0, "#ffffff");
        halo.addColorStop(0.18, baseColor);
        halo.addColorStop(0.62, `${baseColor}a8`);
        halo.addColorStop(1, `${baseColor}00`);
        context.fillStyle = halo;
        context.beginPath();
        context.arc(item.x, item.y, radius * 1.8, 0, Math.PI * 2);
        context.fill();

        if (isSelected) {
          const pulse =
            radius * (2.2 + Math.sin(timestamp / 360) * 0.32);
          context.strokeStyle = "rgba(101, 244, 195, 0.6)";
          context.lineWidth = 1.2;
          context.beginPath();
          context.arc(item.x, item.y, pulse, 0, Math.PI * 2);
          context.stroke();
        }
        context.restore();

        const shouldLabel =
          point.type !== "DiseaseCase" ||
          isSelected ||
          (isActive && positioned.length < 18);
        if (shouldLabel) {
          context.save();
          context.globalAlpha = isActive ? 0.96 : 0.35;
          context.font =
            point.type === "DiseaseCase"
              ? "10px Inter, sans-serif"
              : "500 11px Inter, sans-serif";
          context.textAlign = "center";
          context.fillStyle = "#e9f5f1";
          context.fillText(
            point.label.slice(0, 18),
            item.x,
            item.y + radius + 15,
          );
          if (typeof point.score === "number") {
            context.font = "10px ui-monospace, monospace";
            context.fillStyle = "#8eb8ac";
            context.fillText(
              point.score.toFixed(4),
              item.x,
              item.y + radius + 28,
            );
          }
          context.restore();
        }
      }

      frame = window.requestAnimationFrame(draw);
    }

    function pointerMove(event: PointerEvent) {
      if (!canvas) return;
      const bounds = canvas.getBoundingClientRect();
      const interaction = interactionRef.current;
      interaction.pointerX =
        (event.clientX - bounds.left) / bounds.width - 0.5;
      interaction.pointerY =
        (event.clientY - bounds.top) / bounds.height - 0.5;
      if (interaction.dragging) {
        interaction.manualYaw +=
          (event.clientX - interaction.lastX) * 0.006;
        interaction.manualPitch +=
          (event.clientY - interaction.lastY) * 0.004;
        interaction.lastX = event.clientX;
        interaction.lastY = event.clientY;
      }
    }

    function pointerDown(event: PointerEvent) {
      const interaction = interactionRef.current;
      interaction.dragging = true;
      interaction.lastX = event.clientX;
      interaction.lastY = event.clientY;
      canvas?.setPointerCapture(event.pointerId);
    }

    function pointerUp(event: PointerEvent) {
      interactionRef.current.dragging = false;
      if (canvas?.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
    }

    canvas.addEventListener("pointermove", pointerMove);
    canvas.addEventListener("pointerdown", pointerDown);
    canvas.addEventListener("pointerup", pointerUp);
    canvas.addEventListener("pointercancel", pointerUp);
    frame = window.requestAnimationFrame(draw);

    return () => {
      window.cancelAnimationFrame(frame);
      observer.disconnect();
      canvas.removeEventListener("pointermove", pointerMove);
      canvas.removeEventListener("pointerdown", pointerDown);
      canvas.removeEventListener("pointerup", pointerUp);
      canvas.removeEventListener("pointercancel", pointerUp);
    };
  }, [
    activeRelation,
    activeSet,
    edges,
    mode,
    positioned,
    selectedId,
  ]);

  return (
    <canvas
      className="spatial-scene"
      ref={canvasRef}
      role="img"
      aria-label={ariaLabel}
    />
  );
}
