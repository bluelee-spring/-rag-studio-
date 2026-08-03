"use client";

import type { CSSProperties } from "react";

import type { TraceStage } from "@/lib/types";
import {
  DEEP_STAGE_KINDS,
  DeepStageRenderer,
} from "./DeepVisuals";
import { GraphView } from "./GraphView";

type JsonRecord = Record<string, unknown>;

function asRecords(value: unknown): JsonRecord[] {
  return Array.isArray(value)
    ? value.filter(
        (item): item is JsonRecord =>
          typeof item === "object" && item !== null,
      )
    : [];
}

function asStrings(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String) : [];
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function delay(index: number): CSSProperties {
  return {
    "--delay": `${index * 90}ms`,
    "--index": index,
  } as CSSProperties;
}

function RawData({ data }: { data: JsonRecord }) {
  return (
    <details className="raw-data">
      <summary>查看原始数据</summary>
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </details>
  );
}

function TokenVisual({ data }: { data: JsonRecord }) {
  const tokens = asStrings(data.tokens);
  const normalizations = asRecords(data.normalizations);
  return (
    <div className="token-visual">
      <div className="sentence-source">
        <span>用户问题</span>
        <div className="sentence-line">
          {tokens.map((token, index) => (
            <b key={`${token}-${index}`} style={delay(index)}>
              {token}
            </b>
          ))}
        </div>
      </div>
      <div className="flow-arrow">
        <i />
        <span>拆分</span>
      </div>
      <div className="token-machine">
        <div className="machine-slot" />
        <strong>查询词</strong>
        <div className="token-output">
          {tokens.map((token, index) => (
            <span key={`${token}-out-${index}`} style={delay(index)}>
              {token}
            </span>
          ))}
        </div>
      </div>
      {normalizations.length > 0 && (
        <div className="normalization-track">
          {normalizations.map((item, index) => (
            <span key={index}>
              {formatValue(item.source)}
              <i>→</i>
              <b>{formatValue(item.canonical)}</b>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function SparseVectorVisual({ data }: { data: JsonRecord }) {
  const terms = asRecords(data.terms);
  const weights = terms.map((item) =>
    Math.abs(Number(item.normalized ?? item.idf ?? item.weight ?? 0)),
  );
  const max = Math.max(...weights, 0.0001);
  return (
    <div className="sparse-visual">
      <div className="vector-axis">
        <span className="axis-label">稀疏查询向量</span>
        <div className="vector-columns">
          {terms.map((item, index) => {
            const value = weights[index];
            const height = Math.max(8, (value / max) * 100);
            return (
              <div className="vector-column" key={String(item.term ?? index)}>
                <div className="column-space">
                  <i
                    style={
                      {
                        "--bar-height": `${height}%`,
                        "--delay": `${index * 110}ms`,
                      } as CSSProperties
                    }
                  />
                </div>
                <strong>{formatValue(item.term)}</strong>
                <small>{value.toFixed(2)}</small>
              </div>
            );
          })}
        </div>
        <div className="axis-line" />
      </div>
      <div className="math-caption">
        <span>{formatValue(data.formula)}</span>
        {Boolean(data.parameters) && (
          <b>
            k1={formatValue((data.parameters as JsonRecord).k1)} · b=
            {formatValue((data.parameters as JsonRecord).b)}
          </b>
        )}
      </div>
    </div>
  );
}

function TextToVectorVisual({ data }: { data: JsonRecord }) {
  const text = String(data.text ?? "");
  const characters = Array.from(text).filter((item) => item.trim()).slice(0, 28);
  return (
    <div className="text-vector-visual">
      <div className="text-capsule">
        <span>自然语言</span>
        <p>{text}</p>
      </div>
      <div className="embedding-core">
        <div className="core-rings">
          <i />
          <i />
          <b>E</b>
        </div>
        <small>{formatValue(data.provider)}</small>
      </div>
      <div className="character-field" aria-label="文本被编码为分布式特征">
        {characters.map((character, index) => (
          <i key={`${character}-${index}`} style={delay(index)}>
            {character}
          </i>
        ))}
      </div>
    </div>
  );
}

function DenseVectorVisual({ data }: { data: JsonRecord }) {
  const values = Array.isArray(data.values)
    ? data.values.map(Number).slice(0, 96)
    : [];
  const max = Math.max(...values.map((value) => Math.abs(value)), 0.0001);
  return (
    <div className="dense-vector-visual">
      <div className="query-orb">
        <span>Q</span>
        <i />
      </div>
      <div className="vector-beam">
        {values.map((value, index) => {
          const strength = Math.abs(value) / max;
          return (
            <i
              key={index}
              style={
                {
                  "--strength": strength,
                  "--delay": `${index * 12}ms`,
                } as CSSProperties
              }
              className={value >= 0 ? "positive" : "negative"}
            />
          );
        })}
      </div>
      <div className="dimension-mark">
        <strong>{formatValue(data.dimensions)}维</strong>
        <small>每一格不再对应一个词</small>
      </div>
    </div>
  );
}

function VectorIndexVisual({ data }: { data: JsonRecord }) {
  const dots = Array.from({ length: 54 });
  return (
    <div className="faiss-visual">
      <div className="index-input">
        <span>228 个文本块</span>
        <div className="mini-docs">
          {Array.from({ length: 7 }).map((_, index) => (
            <i
              key={index}
              style={
                {
                  ...delay(index),
                  "--stack-x": `${index * 4}px`,
                  "--stack-y": `${index * 5}px`,
                } as CSSProperties
              }
            />
          ))}
        </div>
      </div>
      <div className="index-stream">
        {Array.from({ length: 5 }).map((_, index) => (
          <i key={index} style={delay(index)} />
        ))}
      </div>
      <div className="faiss-cylinder">
        <div className="cylinder-top">
          <span>{formatValue(data.index_type)}</span>
        </div>
        <div className="vector-dots">
          {dots.map((_, index) => (
            <i key={index} style={delay(index % 12)} />
          ))}
        </div>
        <div className="cylinder-bottom" />
      </div>
      <div className="query-probe">
        <i />
        <span>查询向量</span>
        <b>Top-K</b>
      </div>
      <div className="faiss-caption">
        <span>{formatValue(data.vector_count)} 条向量</span>
        <span>{formatValue(data.dimensions)} 维</span>
        <span>{formatValue(data.search_mode)} search</span>
      </div>
    </div>
  );
}

function RankingVisual({ data }: { data: JsonRecord }) {
  const results = asRecords(data.results);
  const max = Math.max(
    ...results.map((item) => Number(item.score ?? 0)),
    0.0001,
  );
  return (
    <div className="retrieval-arena">
      <div className="search-origin">
        <div className="origin-orb">Q</div>
        <span>查询</span>
      </div>
      <div className="search-field">
        <div className="radar-wave" />
        <div className="radar-wave second" />
        <i className="search-ray" />
      </div>
      <div className="candidate-stack">
        {results.map((result, index) => {
          const score = Number(result.score ?? 0);
          return (
            <div
              className={index === 0 ? "candidate-card winner" : "candidate-card"}
              key={String(result.id ?? index)}
              style={
                {
                  "--score": score / max,
                  "--delay": `${index * 150}ms`,
                } as CSSProperties
              }
            >
              <span>{index === 0 ? "最近" : `#${index + 1}`}</span>
              <div>
                <strong>
                  {formatValue(result.title ?? result.name ?? result.id)}
                </strong>
                <p>{formatValue(result.excerpt).slice(0, 84)}</p>
              </div>
              <b>{score.toFixed(3)}</b>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function CodeVisual({ data }: { data: JsonRecord }) {
  const language = String(data.language ?? "query").toUpperCase();
  const schemaLabels =
    language === "SQL"
      ? ["field_case", "case_symptom", "disease"]
      : language === "SPARQL"
        ? ["DiseaseCase", "hasSymptom", "Pesticide"]
        : ["DiseaseCase", "hasObservedSymptom", "Disease"];
  return (
    <div className="code-visual">
      <div className="schema-stack">
        <span>允许的数据结构</span>
        {schemaLabels.map((label, index) => (
          <div key={label} style={delay(index)}>
            <i />
            <strong>{label}</strong>
          </div>
        ))}
      </div>
      <div className="code-builder">
        <span>{language}</span>
        <div className="builder-gears">
          <i>⌁</i>
          <i>⌁</i>
        </div>
        <small>受约束生成</small>
      </div>
      <div className="query-terminal">
        <header>
          <i />
          <i />
          <i />
          <span>只读查询</span>
        </header>
        <pre>
          <code>{String(data.code ?? "").slice(0, 720)}</code>
        </pre>
      </div>
      <details className="raw-data code-detail">
        <summary>展开完整查询与参数</summary>
        <pre>{String(data.code ?? "")}</pre>
        {Boolean(data.parameters) && (
          <pre>{JSON.stringify(data.parameters, null, 2)}</pre>
        )}
      </details>
    </div>
  );
}

function AnchorVisual({ data }: { data: JsonRecord }) {
  const candidates = asRecords(data.candidates);
  const selected = String(data.selected ?? candidates[0]?.id ?? "");
  return (
    <div className="anchor-visual">
      <div className="anchor-phrase">
        <span>问题中的描述</span>
        <strong>{formatValue(data.query)}</strong>
      </div>
      <div className="anchor-beams">
        {candidates.map((candidate, index) => (
          <i
            key={index}
            style={
              {
                ...delay(index),
                "--beam-offset": `${(index - 1.5) * 48}px`,
                "--beam-angle": `${(index - 1.5) * 8}deg`,
              } as CSSProperties
            }
          />
        ))}
      </div>
      <div className="entity-candidates">
        {candidates.map((candidate, index) => (
          <div
            key={String(candidate.id ?? index)}
            className={
              String(candidate.id) === selected || index === 0
                ? "entity-node selected"
                : "entity-node"
            }
            style={delay(index)}
          >
            <i />
            <strong>{formatValue(candidate.name)}</strong>
            <small>{Number(candidate.score ?? 0).toFixed(3)}</small>
          </div>
        ))}
      </div>
    </div>
  );
}

function RouteVisual({ data }: { data: JsonRecord }) {
  const routes = asRecords(data.routes);
  return (
    <div className="route-visual">
      <div className="route-question">
        <span>复合问题</span>
        <p>{formatValue(data.question)}</p>
      </div>
      <div className="split-junction">
        <i />
        <b />
      </div>
      <div className="route-lanes">
        {routes.map((route, index) => (
          <div key={index} style={delay(index)}>
            <span>{String(index + 1).padStart(2, "0")}</span>
            <i />
            <strong>{formatValue(route.action)}</strong>
            <small>{formatValue(route.executor)}</small>
          </div>
        ))}
      </div>
      <div className="merge-context">
        <i />
        <span>统一证据上下文</span>
      </div>
    </div>
  );
}

function PlanVisual({ data }: { data: JsonRecord }) {
  const plan = (data.plan as JsonRecord | undefined) ?? data;
  const entries = Object.entries(plan)
    .filter(([, value]) => typeof value !== "object")
    .slice(0, 8);
  return (
    <div className="plan-visual">
      <div className="plan-question-mark">?</div>
      <div className="plan-parser">
        <i />
        <strong>结构化参数</strong>
      </div>
      <div className="plan-slots">
        {entries.map(([key, value], index) => (
          <div key={key} style={delay(index)}>
            <span>{key}</span>
            <strong>{formatValue(value)}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function SqlResultVisual({ data }: { data: JsonRecord }) {
  const diseases = asRecords(data.disease_counts);
  const companions = asRecords(data.companion_symptoms).slice(0, 4);
  const matched = Number(data.matched_case_count ?? 0);
  const mainCount = Number(diseases[0]?.case_count ?? 0);
  const pesticide = data.pesticide as JsonRecord | undefined;
  return (
    <div className="sql-result-visual">
      <div className="join-machine">
        {[
          ["field_case", "case_id"],
          ["case_symptom", "case_id"],
          ["disease", "disease_id"],
        ].map(([table, key], index) => (
          <div className="mini-table" key={table} style={delay(index)}>
            <strong>{table}</strong>
            <span>{key}</span>
            <i />
            <i />
          </div>
        ))}
        <div className="join-core">
          <span>JOIN</span>
          <i />
        </div>
      </div>
      <div className="case-cluster" aria-label={`${matched}个匹配病例`}>
        {Array.from({ length: Math.min(matched, 32) }).map((_, index) => (
          <i
            key={index}
            className={index < mainCount ? "main-case" : "other-case"}
            style={delay(index)}
          />
        ))}
        <div>
          <strong>{matched} 个病例</strong>
          <span>
            其中 {formatValue(diseases[0]?.disease_name)} {mainCount} 例
          </span>
        </div>
      </div>
      <div className="fact-ribbon">
        {companions.map((item, index) => (
          <span key={index}>
            {formatValue(item.symptom_name)}
            <b>{formatValue(item.case_count)}例</b>
          </span>
        ))}
        {pesticide && (
          <span className="pesticide-fact">
            {formatValue(pesticide.pesticide_name ?? pesticide.pesticideName)}
            <b>
              {formatValue(
                pesticide.safe_interval_days ?? pesticide.safeIntervalDays,
              )}
              天
            </b>
          </span>
        )}
      </div>
      <RawData data={data} />
    </div>
  );
}

function QueryPlanVisual({ data }: { data: JsonRecord }) {
  const rows = asRecords(data.rows);
  return (
    <div className="query-plan-visual">
      <div className="plan-track">
        {rows.slice(0, 8).map((row, index) => (
          <div key={index} style={delay(index)}>
            <span>{String(index + 1).padStart(2, "0")}</span>
            <i />
            <strong>{formatValue(row.detail)}</strong>
          </div>
        ))}
      </div>
      <div className="plan-output">
        <i />
        <span>SQLite 执行器</span>
      </div>
    </div>
  );
}

function OntologyVisual({ data }: { data: JsonRecord }) {
  const properties = asRecords(data.object_properties).slice(0, 6);
  return (
    <div className="ontology-visual">
      <div className="triple-title">
        <span>主语</span>
        <span>谓语</span>
        <span>宾语</span>
      </div>
      <div className="triple-lanes">
        {properties.map((item, index) => (
          <div key={index} style={delay(index)}>
            <strong>{formatValue(item.domain)}</strong>
            <i />
            <b>{formatValue(item.id)}</b>
            <i />
            <strong>{formatValue(item.range)}</strong>
          </div>
        ))}
      </div>
      <div className="ontology-vocabulary">
        {asRecords(data.classes)
          .slice(0, 8)
          .map((item, index) => (
            <span key={index}>{formatValue(item.id)}</span>
          ))}
      </div>
    </div>
  );
}

function ContextVisual({ data }: { data: JsonRecord }) {
  return (
    <div className="context-visual">
      <div className="evidence-streams">
        {["SQL 精确计数", "属性图邻域", "RDF 约束证据"].map(
          (item, index) => (
            <div key={item} style={delay(index)}>
              <i />
              <span>{item}</span>
              <b>→</b>
            </div>
          ),
        )}
      </div>
      <div className="context-window">
        <header>CONTEXT</header>
        <p>{formatValue(data.policy)}</p>
        <span>{formatValue(data.conflict_strategy)}</span>
      </div>
      <div className="context-to-model">
        <i />
        <b>LLM</b>
      </div>
    </div>
  );
}

function GenerationVisual({ data }: { data: JsonRecord }) {
  const layers = asRecords(data.prompt_layers);
  const evidence = asRecords(data.evidence_cards);
  const fragments = asStrings(data.output_fragments);
  const llmEnabled = Boolean(data.llm_enabled);
  return (
    <div className="generation-visual">
      <div className="prompt-stack">
        <header>
          <span>上下文窗口</span>
          <small>{formatValue(data.context_characters)} 字符</small>
        </header>
        {layers.map((layer, index) => (
          <div
            className={`prompt-layer role-${String(layer.role)}`}
            key={index}
            style={delay(index)}
          >
            <span>{formatValue(layer.label)}</span>
            <p>{formatValue(layer.content)}</p>
          </div>
        ))}
        <div className="evidence-miniatures">
          {evidence.slice(0, 4).map((item, index) => (
            <i key={index} style={delay(index)}>
              {String(item.id ?? index + 1).slice(-4)}
            </i>
          ))}
        </div>
      </div>
      <div className="context-particles">
        {Array.from({ length: 7 }).map((_, index) => (
          <i key={index} style={delay(index)} />
        ))}
      </div>
      <div className={llmEnabled ? "llm-core active" : "llm-core fallback"}>
        <div className="llm-orbit">
          <i />
          <i />
          <i />
        </div>
        <strong>LLM</strong>
        <span>{formatValue(data.provider)}</span>
        <small>{llmEnabled ? "证据约束生成" : "本地模板代替"}</small>
      </div>
      <div className="token-output-visual">
        <header>
          <i />
          <span>回答逐段生成</span>
        </header>
        <p>
          {fragments.map((fragment, index) => (
            <span key={index} style={delay(index)}>
              {fragment}
            </span>
          ))}
          <i className="typing-caret" />
        </p>
      </div>
    </div>
  );
}

export function StageRenderer({ stage }: { stage: TraceStage }) {
  const data = stage.data as JsonRecord;

  if (DEEP_STAGE_KINDS.has(stage.kind)) {
    return <DeepStageRenderer stage={stage} />;
  }

  if (stage.kind === "tokens") return <TokenVisual data={data} />;
  if (stage.kind === "sparse-vector")
    return <SparseVectorVisual data={data} />;
  if (stage.kind === "text") return <TextToVectorVisual data={data} />;
  if (stage.kind === "dense-vector")
    return <DenseVectorVisual data={data} />;
  if (stage.kind === "vector-index")
    return <VectorIndexVisual data={data} />;
  if (stage.kind === "ranking") return <RankingVisual data={data} />;
  if (stage.kind === "code") return <CodeVisual data={data} />;
  if (stage.kind === "anchor") return <AnchorVisual data={data} />;
  if (stage.kind === "route") return <RouteVisual data={data} />;
  if (stage.kind === "plan") return <PlanVisual data={data} />;
  if (stage.kind === "table") return <SqlResultVisual data={data} />;
  if (stage.kind === "query-plan")
    return <QueryPlanVisual data={data} />;
  if (stage.kind === "ontology") return <OntologyVisual data={data} />;
  if (stage.kind === "context") return <ContextVisual data={data} />;
  if (stage.kind === "generation")
    return <GenerationVisual data={data} />;

  if (stage.kind === "graph") {
    const graph = data.graph as
      | {
          nodes: { id: string; label: string; type: string }[];
          edges: { source: string; target: string; relation: string }[];
        }
      | undefined;
    return (
      <div className="graph-stage">
        {graph && <GraphView graph={graph} />}
        <div className="graph-facts">
          {asRecords(data.disease_counts)
            .slice(0, 3)
            .map((item, index) => (
              <span key={index}>
                {formatValue(item.disease_name)}
                <b>{formatValue(item.case_count)}例</b>
              </span>
            ))}
          {asRecords(data.companion_symptoms)
            .slice(0, 3)
            .map((item, index) => (
              <span key={`symptom-${index}`}>
                {formatValue(item.symptom_name)}
                <b>{formatValue(item.case_count)}例</b>
              </span>
            ))}
        </div>
      </div>
    );
  }

  return <RawData data={data} />;
}
