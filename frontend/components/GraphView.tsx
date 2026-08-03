"use client";

import type { CSSProperties } from "react";

type GraphNode = {
  id: string;
  label: string;
  type: string;
};

type GraphEdge = {
  source: string;
  target: string;
  relation: string;
};

type Point = GraphNode & { x: number; y: number };

const colors: Record<string, string> = {
  DiseaseCase: "#C9D8D1",
  Symptom: "#B8C9DF",
  Disease: "#D8C4A7",
  Region: "#D8C5C2",
  Pesticide: "#CBBFDC",
  Document: "#E0D7B8",
  Crop: "#B9D0B5",
  Resource: "#CCD1D4",
};

function layout(nodes: GraphNode[]): Point[] {
  const grouped = new Map<string, GraphNode[]>();
  for (const node of nodes) {
    const group = grouped.get(node.type) ?? [];
    group.push(node);
    grouped.set(node.type, group);
  }

  const points: Point[] = [];
  for (const [type, values] of grouped) {
    if (type === "DiseaseCase") {
      values.forEach((node, index) => {
        points.push({
          ...node,
          x: 190 + (index % 7) * 58,
          y: 76 + Math.floor(index / 7) * 78,
        });
      });
      continue;
    }

    const columns: Record<string, number> = {
      Crop: 70,
      Region: 70,
      Disease: 565,
      Symptom: 705,
      Pesticide: 810,
      Document: 865,
      Resource: 760,
    };
    const x = columns[type] ?? 760;
    values.forEach((node, index) => {
      const spacing = Math.min(82, 320 / Math.max(values.length, 1));
      points.push({
        ...node,
        x,
        y: 62 + index * spacing,
      });
    });
  }
  return points;
}

export function GraphView({
  graph,
}: {
  graph: { nodes: GraphNode[]; edges: GraphEdge[] };
}) {
  const nodes = layout(graph.nodes ?? []);
  const positions = new Map(nodes.map((node) => [node.id, node]));

  if (!nodes.length) {
    return <div className="empty-panel">没有可显示的子图</div>;
  }

  return (
    <div className="graph-wrap">
      <svg
        className="graph-canvas"
        viewBox="0 0 920 420"
        role="img"
        aria-label="检索得到的局部知识图"
      >
        <defs>
          <marker
            id="arrow"
            markerWidth="7"
            markerHeight="7"
            refX="7"
            refY="3.5"
            orient="auto"
          >
            <polygon points="0 0, 7 3.5, 0 7" fill="#A7AFAC" />
          </marker>
        </defs>
        {graph.edges.map((edge, index) => {
          const source = positions.get(edge.source);
          const target = positions.get(edge.target);
          if (!source || !target) return null;
          return (
            <g key={`${edge.source}-${edge.relation}-${edge.target}-${index}`}>
              <line
                className="graph-edge-line"
                x1={source.x}
                y1={source.y}
                x2={target.x}
                y2={target.y}
                stroke="#B9C0BD"
                strokeWidth="1.2"
                markerEnd="url(#arrow)"
                style={
                  {
                    "--edge-delay": `${Math.min(index, 18) * 80}ms`,
                  } as CSSProperties
                }
              >
                <title>{edge.relation}</title>
              </line>
              {index < 12 && (
                <circle className="graph-traversal-dot" r="3">
                  <animateMotion
                    path={`M ${source.x} ${source.y} L ${target.x} ${target.y}`}
                    dur={`${1.7 + (index % 4) * 0.3}s`}
                    begin={`${index * 0.13}s`}
                    repeatCount="indefinite"
                  />
                </circle>
              )}
            </g>
          );
        })}
        {nodes.map((node, index) => {
          const isCase = node.type === "DiseaseCase";
          const label = isCase
            ? node.id.replace("CASE-2025-", "#")
            : node.label;
          return (
            <g
              className="graph-node-enter"
              key={node.id}
              transform={`translate(${node.x},${node.y})`}
              style={
                {
                  "--node-delay": `${Math.min(index, 24) * 55}ms`,
                } as CSSProperties
              }
            >
              <circle
                r={isCase ? 13 : 20}
                fill={colors[node.type] ?? colors.Resource}
                stroke="#FFFFFF"
                strokeWidth="2"
              />
              <text
                y={isCase ? 4 : 3}
                textAnchor="middle"
                className={isCase ? "graph-case-text" : "graph-node-mark"}
              >
                {isCase ? label : node.type.slice(0, 1)}
              </text>
              {!isCase && (
                <text
                  y="34"
                  textAnchor="middle"
                  className="graph-node-label"
                >
                  {node.label.length > 11
                    ? `${node.label.slice(0, 11)}…`
                    : node.label}
                </text>
              )}
              <title>
                {node.id} · {node.label} · {node.type}
              </title>
            </g>
          );
        })}
      </svg>
      <div className="graph-legend">
        {Array.from(new Set(nodes.map((node) => node.type))).map((type) => (
          <span key={type}>
            <i style={{ background: colors[type] ?? colors.Resource }} />
            {type}
          </span>
        ))}
      </div>
    </div>
  );
}
