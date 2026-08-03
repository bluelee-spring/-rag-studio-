"use client";

import {
  CSSProperties,
  ReactElement,
  useEffect,
  useMemo,
  useState,
} from "react";

import type { TraceStage } from "@/lib/types";
import {
  SpatialEdge,
  SpatialPoint,
  SpatialScene,
} from "./SpatialScene";

type JsonRecord = Record<string, unknown>;

export const DEEP_STAGE_KINDS = new Set([
  "tokens",
  "vocabulary-space",
  "tfidf-build",
  "cosine-workbench",
  "bm25-corpus",
  "bm25-document",
  "bm25-accumulator",
  "text",
  "vector-index",
  "dense-vector",
  "vector-space",
  "dense-similarity",
  "entity-space",
  "graph-pattern",
  "graph-traversal",
  "graph-aggregate",
  "ontology-space",
  "iri-mapping",
  "triple-pattern",
  "rdf-filter",
  "relational-plan",
  "row-filter",
  "key-join",
  "group-aggregate",
  "table-schema",
  "sql-plan",
  "table-result",
  "ranking",
]);

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

function asNumbers(value: unknown): number[] {
  return Array.isArray(value)
    ? value.map(Number).filter(Number.isFinite)
    : [];
}

function text(value: unknown, fallback = "—"): string {
  if (value === null || value === undefined || value === "") {
    return fallback;
  }
  return String(value);
}

function number(value: unknown, fallback = 0): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function customStyle(
  values: Record<string, string | number>,
): CSSProperties {
  return values as CSSProperties;
}

function FormulaStrip({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="engine-formula">
      <span>{label}</span>
      <code>{value}</code>
    </div>
  );
}

function TokenTrace({ data }: { data: JsonRecord }) {
  const tokens = asStrings(data.tokens);
  const normalizations = asRecords(data.normalizations);
  const oov = new Set(asStrings(data.out_of_vocabulary));
  const question = text(data.question, tokens.join(" · "));

  return (
    <div className="engine-tokenizer">
      <div className="tokenizer-source">
        <span className="engine-label">RAW QUERY</span>
        <p>{question}</p>
        <i className="scan-plane" />
      </div>
      <div className="tokenizer-conveyor" aria-hidden="true">
        <i />
        <i />
        <i />
      </div>
      <div className="tokenizer-ports">
        <span className="engine-label">LONGEST MATCH + NORMALIZE</span>
        <div>
          {tokens.map((token, index) => (
            <span
              className={oov.has(token) ? "token-port oov" : "token-port"}
              key={`${token}-${index}`}
              style={customStyle({
                "--index": index,
                "--delay": `${index * 90}ms`,
              })}
            >
              <i />
              <b>{token}</b>
              <small>{oov.has(token) ? "OOV" : `t${index + 1}`}</small>
            </span>
          ))}
        </div>
      </div>
      {normalizations.length > 0 && (
        <div className="normalization-hud">
          {normalizations.map((item, index) => (
            <span key={index}>
              {text(item.source)}
              <i>→</i>
              <b>{text(item.canonical)}</b>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function VocabularySpace({ data }: { data: JsonRecord }) {
  const vocabulary = asRecords(data.vocabulary);
  const activeTerms = asRecords(data.active_terms);
  const activeIndexes = new Set(
    activeTerms.map((item) => number(item.index)),
  );

  return (
    <div className="vocabulary-space">
      <div className="vocabulary-hud">
        <div>
          <span>固定坐标轴</span>
          <strong>{number(data.dimension_count)} 维</strong>
        </div>
        <i />
        <div>
          <span>文档语料</span>
          <strong>{number(data.document_count)} 块</strong>
        </div>
        <i />
        <div>
          <span>本次激活</span>
          <strong>{activeTerms.length} 维</strong>
        </div>
      </div>

      <div className="dimension-chamber">
        <div className="dimension-floor">
          {vocabulary.map((item, index) => {
            const active = activeIndexes.has(number(item.index));
            return (
              <i
                className={active ? "dimension active" : "dimension"}
                key={text(item.term, String(index))}
                style={customStyle({
                  "--index": index,
                  "--idf": number(item.idf),
                  "--delay": `${(index % 18) * 35}ms`,
                })}
                aria-label={`${text(item.term)}，第${number(item.index)}维`}
              />
            );
          })}
        </div>
        <div className="active-coordinate-labels">
          {activeTerms.slice(0, 12).map((item, index) => (
            <span
              key={text(item.term, String(index))}
              style={customStyle({
                "--index": index,
                "--delay": `${index * 100}ms`,
              })}
            >
              <small>维度 {number(item.index)}</small>
              <b>{text(item.term)}</b>
            </span>
          ))}
        </div>
        <div className="dimension-depth-label">
          <span>[0, 0, …, wᵢ, …, 0]</span>
          <small>词不决定分数，只决定写入哪个固定坐标</small>
        </div>
      </div>
    </div>
  );
}

function VectorRail({
  values,
  activeIndexes,
  label,
  tone = "query",
}: {
  values: number[];
  activeIndexes?: Set<number>;
  label: string;
  tone?: "query" | "document" | "product";
}) {
  const max = Math.max(
    ...values.map((value) => Math.abs(value)),
    0.000001,
  );
  return (
    <div className={`vector-rail tone-${tone}`}>
      <span>{label}</span>
      <div>
        {values.map((value, index) => {
          const active =
            activeIndexes?.has(index) ?? Math.abs(value) > 0.000001;
          return (
            <i
              key={index}
              className={active ? "active" : ""}
              style={customStyle({
                "--value": Math.abs(value) / max,
                "--sign": value >= 0 ? 1 : -1,
                "--delay": `${Math.min(index * 8, 800)}ms`,
              })}
            />
          );
        })}
      </div>
      <small>{values.length} dimensions</small>
    </div>
  );
}

function TfidfBuild({ data }: { data: JsonRecord }) {
  const terms = asRecords(data.terms);
  const rawVector = asNumbers(data.raw_vector);
  const normalizedVector = asNumbers(data.normalized_vector);
  const activeIndexes = new Set(
    terms.map((item) => number(item.index)),
  );

  return (
    <div className="tfidf-engine">
      <div className="tfidf-term-lines">
        <header>
          <span>坐标</span>
          <span>TF</span>
          <span>IDF</span>
          <span>TF × IDF</span>
          <span>归一化</span>
        </header>
        {terms.slice(0, 12).map((item, index) => (
          <div
            className="tfidf-term-line"
            key={text(item.term, String(index))}
            style={customStyle({ "--delay": `${index * 110}ms` })}
          >
            <strong>
              <small>v[{number(item.index)}]</small>
              {text(item.term)}
            </strong>
            <span>{number(item.tf).toFixed(0)}</span>
            <i>×</i>
            <span>{number(item.idf).toFixed(4)}</span>
            <i>=</i>
            <b>{number(item.weight).toFixed(4)}</b>
            <i>÷ ‖q‖</i>
            <em>{number(item.normalized).toFixed(4)}</em>
          </div>
        ))}
      </div>

      <div className="tfidf-vector-stack">
        <VectorRail
          values={rawVector}
          activeIndexes={activeIndexes}
          label="未归一化 q"
        />
        <div className="norm-reactor">
          <span>L2 NORM</span>
          <strong>{number(data.raw_norm).toFixed(5)}</strong>
          <i />
        </div>
        <VectorRail
          values={normalizedVector}
          activeIndexes={activeIndexes}
          label="单位查询向量 q̂"
        />
      </div>
      <FormulaStrip label="WEIGHT" value={text(data.formula)} />
    </div>
  );
}

function CosineWorkbench({ data }: { data: JsonRecord }) {
  const comparisons = asRecords(data.comparisons);
  const [selected, setSelected] = useState(0);
  const comparison = comparisons[selected] ?? comparisons[0];
  const queryVector = asNumbers(comparison?.query_vector);
  const documentVector = asNumbers(comparison?.document_vector);
  const products = asRecords(comparison?.products);
  const activeIndexes = new Set(
    products.map((item) => number(item.index)),
  );
  const productVector = Array.from(
    { length: number(data.dimension_count) },
    () => 0,
  );
  for (const product of products) {
    productVector[number(product.index)] = number(product.product);
  }

  if (!comparison) return <div className="empty-visual">没有候选文档</div>;

  return (
    <div className="cosine-lab">
      <div className="candidate-selector">
        {comparisons.slice(0, 5).map((item, index) => (
          <button
            type="button"
            className={index === selected ? "selected" : ""}
            onClick={() => setSelected(index)}
            key={text(item.id, String(index))}
          >
            <span>#{index + 1}</span>
            <strong>{text(item.title)}</strong>
            <b>{number(item.score).toFixed(4)}</b>
          </button>
        ))}
      </div>

      <div className="cosine-space">
        <VectorRail
          values={queryVector}
          activeIndexes={activeIndexes}
          label="查询单位向量 q̂"
          tone="query"
        />
        <div className="coordinate-couplers" aria-hidden="true">
          {products.map((item, index) => (
            <i
              key={index}
              style={customStyle({
                "--index": index,
                "--delay": `${index * 130}ms`,
              })}
            />
          ))}
          <span>同一坐标对齐相乘</span>
        </div>
        <VectorRail
          values={documentVector}
          activeIndexes={activeIndexes}
          label="文档单位向量 d̂"
          tone="document"
        />
        <VectorRail
          values={productVector}
          activeIndexes={activeIndexes}
          label="逐维乘积 qᵢ × dᵢ"
          tone="product"
        />
      </div>

      <div className="cosine-accumulator">
        <div className="product-stream">
          {products.slice(0, 12).map((item, index) => (
            <span
              key={index}
              style={customStyle({ "--delay": `${index * 90}ms` })}
            >
              <small>{text(item.term)}</small>
              {number(item.query).toFixed(3)}
              <i>×</i>
              {number(item.document).toFixed(3)}
              <b>= {number(item.product).toFixed(4)}</b>
            </span>
          ))}
        </div>
        <div className="sum-core">
          <span>Σ PRODUCTS</span>
          <strong>{number(comparison.numerator).toFixed(6)}</strong>
          <small>
            ÷ {number(comparison.denominator).toFixed(3)}
          </small>
          <b>{number(comparison.score).toFixed(6)}</b>
        </div>
      </div>
      <FormulaStrip label="COSINE" value={text(data.formula)} />
    </div>
  );
}

function Bm25Corpus({ data }: { data: JsonRecord }) {
  const terms = asRecords(data.terms);
  const total = number(data.document_count, 1);

  return (
    <div className="bm25-corpus">
      <div className="corpus-vault">
        <div className="corpus-rings">
          {Array.from({ length: 7 }).map((_, index) => (
            <i key={index} style={customStyle({ "--index": index })} />
          ))}
          <strong>{total}</strong>
          <span>DOCUMENTS</span>
        </div>
      </div>
      <div className="idf-towers">
        {terms.map((item, index) => {
          const ratio = number(item.df) / total;
          const idf = number(item.idf);
          return (
            <div
              className="idf-tower"
              key={text(item.term, String(index))}
              style={customStyle({
                "--ratio": ratio,
                "--idf": Math.min(idf / 4, 1),
                "--delay": `${index * 140}ms`,
              })}
            >
              <div className="document-frequency">
                <i />
                <span>
                  df {number(item.df)} / {total}
                </span>
              </div>
              <div className="idf-lift">
                <i />
                <b>{idf.toFixed(4)}</b>
              </div>
              <strong>{text(item.term)}</strong>
            </div>
          );
        })}
      </div>
      <FormulaStrip label="IDF" value={text(data.idf_formula)} />
    </div>
  );
}

function Bm25Document({ data }: { data: JsonRecord }) {
  const documents = asRecords(data.documents);
  const [selected, setSelected] = useState(0);
  const document = documents[selected] ?? documents[0];
  const terms = asRecords(document?.terms);
  const parameters = (data.parameters as JsonRecord | undefined) ?? {};

  if (!document) return <div className="empty-visual">没有候选文档</div>;

  return (
    <div className="bm25-document-lab">
      <div className="candidate-selector compact">
        {documents.slice(0, 5).map((item, index) => (
          <button
            type="button"
            className={index === selected ? "selected" : ""}
            onClick={() => setSelected(index)}
            key={text(item.id, String(index))}
          >
            <span>D{index + 1}</span>
            <strong>{text(item.title)}</strong>
            <b>{number(item.score).toFixed(3)}</b>
          </button>
        ))}
      </div>

      <div className="bm25-pressure-chamber">
        <div className="document-length-gauge">
          <span>DOCUMENT LENGTH</span>
          <div>
            <i
              style={customStyle({
                "--length-ratio": Math.min(
                  number(document.length_ratio),
                  1.8,
                ),
              })}
            />
            <b>{number(document.length)} tokens</b>
          </div>
          <small>
            avgdl {number(document.average_length).toFixed(2)}
          </small>
        </div>
        <div className="bm25-term-reactors">
          {terms.map((item, index) => (
            <div
              className="bm25-term-reactor"
              key={text(item.term, String(index))}
              style={customStyle({
                "--saturation": Math.min(
                  number(item.saturation) / 2.5,
                  1,
                ),
                "--delay": `${index * 120}ms`,
              })}
            >
              <strong>{text(item.term)}</strong>
              <div className="tf-pressure">
                <span>tf</span>
                <b>{number(item.tf)}</b>
                <i />
              </div>
              <div className="saturation-valve">
                <span>饱和后</span>
                <b>{number(item.saturation).toFixed(4)}</b>
                <i />
              </div>
              <div className="idf-coupling">
                <span>× IDF</span>
                <b>{number(item.idf).toFixed(4)}</b>
              </div>
              <div className="term-contribution">
                <span>贡献</span>
                <b>{number(item.contribution).toFixed(5)}</b>
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="bm25-parameter-strip">
        <span>k1 = {number(parameters.k1)}</span>
        <span>b = {number(parameters.b)}</span>
        <span>dl/avgdl = {number(document.length_ratio).toFixed(3)}</span>
      </div>
      <FormulaStrip label="CONTRIBUTION" value={text(data.formula)} />
    </div>
  );
}

function Bm25Accumulator({ data }: { data: JsonRecord }) {
  const terms = asStrings(data.terms);
  const documents = asRecords(data.documents);
  const maxContribution = Math.max(
    ...documents.flatMap((item) => asNumbers(item.contributions)),
    0.000001,
  );
  const maxScore = Math.max(
    ...documents.map((item) => number(item.score)),
    0.000001,
  );

  return (
    <div className="bm25-accumulator">
      <div
        className="contribution-matrix"
        style={customStyle({ "--columns": terms.length })}
      >
        <header>
          <span>文档</span>
          {terms.map((term) => (
            <b key={term}>{term}</b>
          ))}
          <strong>Σ</strong>
        </header>
        {documents.map((document, rowIndex) => (
          <div className="contribution-row" key={text(document.id)}>
            <span>D{rowIndex + 1}</span>
            {asNumbers(document.contributions).map((value, index) => (
              <i
                key={index}
                style={customStyle({
                  "--energy": value / maxContribution,
                  "--delay": `${(rowIndex * terms.length + index) * 55}ms`,
                })}
              >
                {value.toFixed(3)}
              </i>
            ))}
            <b>{number(document.score).toFixed(4)}</b>
          </div>
        ))}
      </div>
      <div className="score-bus">
        {documents.map((document, index) => (
          <div
            key={text(document.id)}
            style={customStyle({
              "--score": number(document.score) / maxScore,
              "--delay": `${index * 140}ms`,
            })}
          >
            <span>D{index + 1}</span>
            <i />
            <b>{number(document.score).toFixed(4)}</b>
          </div>
        ))}
      </div>
    </div>
  );
}

function SemanticInput({ data }: { data: JsonRecord }) {
  const value = text(data.text);
  const characters = Array.from(value).filter((item) => item.trim());
  return (
    <div className="semantic-input-engine">
      <div className="semantic-query-card">
        <span>NATURAL LANGUAGE</span>
        <p>{value}</p>
      </div>
      <div className="transformer-core">
        <i />
        <i />
        <i />
        <strong>ENCODER</strong>
        <small>{text(data.provider)}</small>
      </div>
      <div className="distributed-features">
        {characters.slice(0, 40).map((character, index) => (
          <i
            key={`${character}-${index}`}
            style={customStyle({
              "--index": index,
              "--delay": `${index * 35}ms`,
              "--feature-depth": `${(index % 4) * 8}px`,
            })}
          >
            {character}
          </i>
        ))}
      </div>
    </div>
  );
}

function VectorIndex({ data }: { data: JsonRecord }) {
  const count = number(data.vector_count);
  return (
    <div className="index-architecture">
      <div className="offline-lane">
        <span className="engine-label">OFFLINE INGESTION</span>
        <div className="document-stack-3d">
          {Array.from({ length: 8 }).map((_, index) => (
            <i key={index} style={customStyle({ "--index": index })} />
          ))}
          <strong>{count} 文本块</strong>
        </div>
        <div className="embedding-gate">
          <i />
          <span>Embedding</span>
        </div>
      </div>
      <div className="faiss-core-3d">
        <div className="faiss-disk top" />
        <div className="faiss-points">
          {Array.from({ length: 72 }).map((_, index) => (
            <i
              key={index}
              style={customStyle({
                "--index": index,
                "--delay": `${(index % 14) * 45}ms`,
              })}
            />
          ))}
        </div>
        <div className="faiss-disk bottom" />
        <strong>{text(data.index_type)}</strong>
        <span>
          {number(data.dimensions)}D · {count} vectors
        </span>
      </div>
      <div className="online-lane">
        <span className="engine-label">ONLINE QUERY</span>
        <div className="query-probe-3d">
          <i />
          <b>q</b>
        </div>
        <div className="topk-return">
          <i />
          <span>positions + scores</span>
        </div>
      </div>
      <div className="index-storage-note">
        <span>{text(data.index_file)}</span>
        <i />
        <span>{text(data.metadata_file)}</span>
      </div>
    </div>
  );
}

function DenseVector({ data }: { data: JsonRecord }) {
  const values = asNumbers(data.values);
  const max = Math.max(
    ...values.map((value) => Math.abs(value)),
    0.000001,
  );
  return (
    <div className="dense-vector-chamber">
      <div className="dense-axis-cloud">
        {values.map((value, index) => (
          <i
            key={index}
            className={value >= 0 ? "positive" : "negative"}
            style={customStyle({
              "--magnitude": Math.abs(value) / max,
              "--index": index,
              "--delay": `${Math.min(index * 8, 900)}ms`,
            })}
          >
            <span>{index}</span>
          </i>
        ))}
      </div>
      <div className="dense-vector-orb">
        <span>‖q‖₂</span>
        <strong>{number(data.norm).toFixed(5)}</strong>
        <i />
      </div>
      <div className="dense-dimension-note">
        <strong>{number(data.dimensions)} dimensions</strong>
        <span>单维无固定词义 · 整体位置表达语义</span>
      </div>
    </div>
  );
}

function VectorSpace({ data }: { data: JsonRecord }) {
  const query = (data.query as JsonRecord | undefined) ?? {};
  const queryProjection =
    (query.projection as JsonRecord | undefined) ?? {};
  const points: SpatialPoint[] = [
    {
      id: "QUERY",
      label: "QUERY",
      type: "Query",
      x: number(queryProjection.x),
      y: number(queryProjection.y),
      z: number(queryProjection.z),
    },
    ...asRecords(data.points).map((item) => {
      const projection =
        (item.projection as JsonRecord | undefined) ?? {};
      return {
        id: text(item.id),
        label: text(item.title),
        type: "Document",
        x: number(projection.x),
        y: number(projection.y),
        z: number(projection.z),
        score: number(item.score),
      };
    }),
  ];
  const selected = points[1]?.id;

  return (
    <div className="spatial-stage-shell">
      <SpatialScene
        points={points}
        selectedId={selected}
        mode="embedding"
        ariaLabel={`${number(data.dimensions)}维语义向量的空间投影，查询点与Top-K文档点`}
      />
      <div className="spatial-stage-hud">
        <span>{number(data.vector_count)} vectors in index</span>
        <b>{number(data.dimensions)}D exact inner product</b>
        <small>{text(data.projection_note)}</small>
      </div>
      <div className="scene-gesture">拖动旋转空间</div>
    </div>
  );
}

function DenseSimilarity({ data }: { data: JsonRecord }) {
  const comparisons = asRecords(data.comparisons);
  const [selected, setSelected] = useState(0);
  const comparison = comparisons[selected] ?? comparisons[0];
  const groups = asRecords(comparison?.groups);
  const maxProduct = Math.max(
    ...groups.map((item) => Math.abs(number(item.product))),
    0.000001,
  );

  if (!comparison) return <div className="empty-visual">没有候选向量</div>;

  return (
    <div className="dense-similarity-lab">
      <div className="candidate-selector compact">
        {comparisons.slice(0, 5).map((item, index) => (
          <button
            type="button"
            className={selected === index ? "selected" : ""}
            onClick={() => setSelected(index)}
            key={text(item.id)}
          >
            <span>#{index + 1}</span>
            <strong>{text(item.title)}</strong>
            <b>{number(item.score).toFixed(4)}</b>
          </button>
        ))}
      </div>
      <div className="dense-product-tunnel">
        {groups.map((group, index) => (
          <div
            className={
              number(group.product) >= 0
                ? "product-group positive"
                : "product-group negative"
            }
            key={index}
            style={customStyle({
              "--energy":
                Math.abs(number(group.product)) / maxProduct,
              "--delay": `${index * 55}ms`,
            })}
          >
            <span>
              {number(group.from)}–{number(group.to)}
            </span>
            <i />
            <b>{number(group.product).toFixed(5)}</b>
          </div>
        ))}
        <div className="dense-sum-reactor">
          <span>Σ 192 PRODUCTS</span>
          <strong>{number(comparison.score).toFixed(6)}</strong>
          <small>
            ‖q‖={number(comparison.query_norm).toFixed(3)} · ‖d‖=
            {number(comparison.document_norm).toFixed(3)}
          </small>
        </div>
      </div>
    </div>
  );
}

function EntitySpace({ data }: { data: JsonRecord }) {
  const candidates = asRecords(data.candidates);
  const queryProjection =
    (data.query_projection as JsonRecord | undefined) ?? {};
  const points: SpatialPoint[] = [
    {
      id: "QUERY",
      label: text(data.query_label, "用户问题"),
      type: "Query",
      x: number(queryProjection.x),
      y: number(queryProjection.y),
      z: number(queryProjection.z),
    },
    ...candidates.map((item) => {
      const projection =
        (item.projection as JsonRecord | undefined) ?? {};
      return {
        id: text(item.id),
        label: text(item.name),
        type: text(item.type, "Entity"),
        x: number(projection.x),
        y: number(projection.y),
        z: number(projection.z),
        score: number(item.score),
      };
    }),
  ];
  const selectedId = text(data.selected);
  const selected = candidates.find(
    (item) => text(item.id) === selectedId,
  );
  const productGroups = asRecords(selected?.product_groups);

  return (
    <div className="entity-space-layout">
      <div className="spatial-stage-shell">
        <SpatialScene
          points={points}
          selectedId={selectedId}
          mode="embedding"
          ariaLabel="用户问题与图节点语义向量的空间投影及最近实体锚点"
        />
        <div className="scene-gesture">拖动旋转空间</div>
      </div>
      <div className="anchor-inspector">
        <span className="engine-label">FULL-DIMENSION SIMILARITY</span>
        <strong>{text(selected?.name)}</strong>
        <b>{number(selected?.score).toFixed(6)}</b>
        <div className="anchor-product-spectrum">
          {productGroups.map((group, index) => (
            <i
              key={index}
              style={customStyle({
                "--product": Math.min(
                  Math.abs(number(group.product)) * 20,
                  1,
                ),
                "--delay": `${index * 60}ms`,
              })}
            />
          ))}
        </div>
        <small>
          {number(data.dimensions)}维内积选择锚点；三维位置仅用于观察
        </small>
      </div>
    </div>
  );
}

function GraphPattern({ data }: { data: JsonRecord }) {
  const nodes = asRecords(data.pattern_nodes);
  const edges = asRecords(data.pattern_edges);
  const filters = asRecords(data.filters);
  return (
    <div className="graph-pattern-lab">
      <div className="pattern-space">
        {nodes.map((node, index) => (
          <div
            className={`pattern-node role-${text(node.role)}`}
            key={text(node.id)}
            style={customStyle({
              "--index": index,
              "--delay": `${index * 150}ms`,
            })}
          >
            <i />
            <span>{text(node.label)}</span>
            {Boolean(node.value) && <small>{text(node.value)}</small>}
          </div>
        ))}
        <svg
          viewBox="0 0 900 360"
          role="img"
          aria-label="病例、地区、症状和疾病组成的允许图模式"
        >
          <defs>
            <linearGradient id="pattern-flow" x1="0" x2="1">
              <stop offset="0" stopColor="#65f4c3" />
              <stop offset="1" stopColor="#72d8ff" />
            </linearGradient>
            <filter id="pattern-glow">
              <feGaussianBlur stdDeviation="5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          {edges.map((edge, index) => {
            const paths = [
              "M450 180 C330 110 245 95 145 86",
              "M450 180 C315 180 230 180 130 180",
              "M450 180 C565 160 650 110 755 88",
            ];
            return (
              <g key={index}>
                <path
                  d={paths[index] ?? paths[0]}
                  className="pattern-edge"
                  style={{ animationDelay: `${index * 180}ms` }}
                />
                <text
                  x={[260, 255, 635][index] ?? 300}
                  y={[100, 169, 112][index] ?? 100}
                >
                  {text(edge.relation)}
                </text>
                <circle r="4" className="pattern-particle">
                  <animateMotion
                    dur={`${1.8 + index * 0.2}s`}
                    repeatCount="indefinite"
                    path={paths[index] ?? paths[0]}
                  />
                </circle>
              </g>
            );
          })}
        </svg>
      </div>
      <div className="pattern-constraints">
        <div>
          <span>FILTERS</span>
          {filters.map((filter, index) => (
            <b key={index}>
              {text(filter.field)} {text(filter.operator)}{" "}
              {text(filter.value)}
            </b>
          ))}
        </div>
        <div>
          <span>TRAVERSAL BOUNDARY</span>
          <b>关系白名单 {asStrings(data.allowed_relationships).length}</b>
          <b>最大跳数 {number(data.max_hops)}</b>
          <small>任意关系和无限变长路径被阻断</small>
        </div>
      </div>
    </div>
  );
}

function graphPayload(data: JsonRecord): {
  points: SpatialPoint[];
  edges: SpatialEdge[];
} {
  const graph = (data.graph as JsonRecord | undefined) ?? {};
  const points = asRecords(graph.nodes).map((node) => ({
    id: text(node.id),
    label: text(node.label),
    type: text(node.type, "Resource"),
  }));
  const edges = asRecords(graph.edges).map((edge) => ({
    source: text(edge.source),
    target: text(edge.target),
    relation: text(edge.relation),
  }));
  return { points, edges };
}

function GraphTraversal({ data }: { data: JsonRecord }) {
  const steps = asRecords(data.steps);
  const [activeStep, setActiveStep] = useState(0);
  const payload = useMemo(() => graphPayload(data), [data]);
  const step = steps[activeStep] ?? steps[0];
  const activeIds = asStrings(step?.active_ids);

  useEffect(() => {
    if (steps.length < 2) return;
    const timer = window.setInterval(() => {
      setActiveStep((current) => (current + 1) % steps.length);
    }, 1450);
    return () => window.clearInterval(timer);
  }, [steps.length]);

  return (
    <div className="graph-traversal-layout">
      <div className="spatial-stage-shell">
        <SpatialScene
          points={payload.points}
          edges={payload.edges}
          selectedId={text(data.anchor_id)}
          activeIds={activeIds}
          activeRelation={text(step?.relation, "") || undefined}
          mode="graph"
          ariaLabel="属性图或RDF图中节点和关系的逐跳遍历过程"
        />
        <div className="scene-gesture">拖动旋转空间</div>
      </div>
      <div className="traversal-console">
        <header>
          <span>TRAVERSAL STEP</span>
          <b>
            {String(activeStep + 1).padStart(2, "0")} /{" "}
            {String(Math.max(steps.length, 1)).padStart(2, "0")}
          </b>
        </header>
        <strong>{text(step?.operation)}</strong>
        {Boolean(step?.relation) && <code>{text(step?.relation)}</code>}
        <div className="traversal-count-flow">
          <span>{number(step?.input_count, 1)}</span>
          <i />
          <b>{number(step?.output_count)}</b>
        </div>
        <div className="traversal-step-dots">
          {steps.map((item, index) => (
            <button
              type="button"
              className={index === activeStep ? "active" : ""}
              onClick={() => setActiveStep(index)}
              aria-label={`查看第${index + 1}步：${text(item.operation)}`}
              key={index}
            >
              {index + 1}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function GraphAggregate({ data }: { data: JsonRecord }) {
  const diseases = asRecords(data.disease_counts);
  const symptoms = asRecords(data.companion_symptoms);
  const total = number(data.matched_case_count);
  const maxDisease = Math.max(
    ...diseases.map((item) => number(item.case_count)),
    1,
  );
  return (
    <div className="graph-aggregate-lab">
      <div className="case-particle-source">
        <span>{total}</span>
        <small>matching cases</small>
        <div>
          {Array.from({ length: Math.min(total, 32) }).map((_, index) => (
            <i
              key={index}
              style={customStyle({
                "--index": index,
                "--orbit-radius": `${50 + (index % 5) * 7}px`,
              })}
            />
          ))}
        </div>
      </div>
      <div className="aggregation-buckets">
        {diseases.map((disease, index) => (
          <div
            key={text(disease.disease_id, String(index))}
            style={customStyle({
              "--ratio": number(disease.case_count) / maxDisease,
              "--delay": `${index * 180}ms`,
            })}
          >
            <i />
            <strong>{text(disease.disease_name)}</strong>
            <b>{number(disease.case_count)}例</b>
          </div>
        ))}
      </div>
      <div className="companion-orbit">
        <span>主要病害病例继续扩展</span>
        <div>
          {symptoms.slice(0, 6).map((symptom, index) => (
            <b
              key={text(symptom.symptom_id, String(index))}
              style={customStyle({
                "--index": index,
                "--delay": `${index * 120}ms`,
                "--orbit-left": `${8 + (index % 3) * 32}%`,
                "--orbit-top": `${Math.floor(index / 3) * 105}px`,
              })}
            >
              {text(symptom.symptom_name)}
              <small>{number(symptom.case_count)}例</small>
            </b>
          ))}
        </div>
      </div>
    </div>
  );
}

function OntologySpace({ data }: { data: JsonRecord }) {
  const classes = asRecords(data.classes).slice(0, 9);
  const properties = asRecords(data.object_properties).slice(0, 8);
  return (
    <div className="ontology-space">
      <div className="ontology-planes">
        <div className="ontology-plane class-plane">
          <span>CLASS LAYER</span>
          {classes.map((item, index) => (
            <i
              key={text(item.id)}
              style={customStyle({
                "--index": index,
                "--delay": `${index * 90}ms`,
              })}
            >
              {text(item.label)}
            </i>
          ))}
        </div>
        <div className="ontology-plane predicate-plane">
          <span>OBJECT PROPERTY LAYER</span>
          {properties.map((item, index) => (
            <i
              key={text(item.id)}
              style={customStyle({
                "--index": index,
                "--delay": `${index * 100}ms`,
              })}
            >
              <small>{text(item.domain)}</small>
              <b>{text(item.id)}</b>
              <small>{text(item.range)}</small>
            </i>
          ))}
        </div>
        <div className="ontology-plane literal-plane">
          <span>DATATYPE PROPERTY LAYER</span>
          {asStrings(data.datatype_properties)
            .slice(0, 9)
            .map((item, index) => (
              <i
                key={item}
                style={customStyle({ "--index": index })}
              >
                {item}
              </i>
            ))}
        </div>
      </div>
    </div>
  );
}

function IriMapping({ data }: { data: JsonRecord }) {
  const mappings = asRecords(data.mappings);
  return (
    <div className="iri-mapping-lab">
      {mappings.map((item, index) => (
        <div
          className="iri-lane"
          key={text(item.local_id, String(index))}
          style={customStyle({ "--delay": `${index * 180}ms` })}
        >
          <div className="surface-name">
            <span>{text(item.role)}</span>
            <strong>{text(item.source)}</strong>
          </div>
          <div className="identity-resolver">
            <i />
            <b>{text(item.local_id)}</b>
          </div>
          <div className="iri-address">
            <span>GLOBAL IRI</span>
            <code>{text(item.iri)}</code>
          </div>
        </div>
      ))}
    </div>
  );
}

function TriplePattern({ data }: { data: JsonRecord }) {
  const nodes = asRecords(data.nodes);
  const nodeIds = new Set(nodes.map((item) => text(item.id)));
  const extraNodes: SpatialPoint[] = [];
  const edges: SpatialEdge[] = asRecords(data.patterns).map(
    (pattern, index) => {
      const object = text(pattern.object);
      if (!nodeIds.has(object)) {
        nodeIds.add(object);
        extraNodes.push({
          id: object,
          label: object,
          type: "iri",
        });
      }
      return {
        source: text(pattern.subject),
        target: object,
        relation: text(pattern.predicate),
      };
    },
  );
  const points: SpatialPoint[] = [
    ...nodes.map((node) => ({
      id: text(node.id),
      label: text(node.label),
      type: text(node.kind),
    })),
    ...extraNodes,
  ];
  const filters = asRecords(data.filters);
  return (
    <div className="triple-pattern-layout">
      <div className="spatial-stage-shell">
        <SpatialScene
          points={points}
          edges={edges}
          mode="graph"
          ariaLabel="SPARQL变量、IRI和字面量组成的三元组查询模式"
        />
        <div className="scene-gesture">拖动旋转空间</div>
      </div>
      <div className="triple-filter-hud">
        <span>VARIABLE BINDING</span>
        {filters.map((filter, index) => (
          <b key={index}>
            {text(filter.variable)} {text(filter.operator)}{" "}
            {text(filter.value)}
          </b>
        ))}
      </div>
    </div>
  );
}

function RdfFilter({ data }: { data: JsonRecord }) {
  const candidates = asRecords(data.candidates);
  const threshold = number(data.threshold);
  const max = Math.max(
    threshold,
    ...candidates.map((item) => number(item.safe_interval_days)),
    1,
  );
  return (
    <div className="rdf-filter-lab">
      <div
        className="literal-axis"
        style={customStyle({
          "--threshold": threshold / max,
        })}
      >
        <span>0 days</span>
        <i />
        <b
          style={customStyle({
            "--threshold": threshold / max,
          })}
        >
          ≤ {threshold}
        </b>
        <span>{max} days</span>
      </div>
      <div className="literal-particles">
        {candidates.map((item, index) => (
          <div
            className={item.passed ? "passed" : "rejected"}
            key={text(item.id, String(index))}
            style={customStyle({
              "--position":
                number(item.safe_interval_days) / max,
              "--delay": `${index * 160}ms`,
              "--row-y": `${index * 52}px`,
            })}
          >
            <i />
            <strong>{text(item.name)}</strong>
            <b>{number(item.safe_interval_days)}天</b>
            <small>{item.passed ? "PASS" : "REJECT"}</small>
          </div>
        ))}
      </div>
      <FormulaStrip
        label="FILTER"
        value={`?${text(data.property)} ${text(data.operator)} ${threshold}`}
      />
    </div>
  );
}

function RelationalPlan({ data }: { data: JsonRecord }) {
  const plan =
    (data.plan as JsonRecord | undefined) ?? {};
  const fields = [
    ["region_id", plan.region_id],
    ["date_start", plan.date_start],
    ["date_end", plan.date_end],
    ["symptom_id", plan.symptom_id],
    ["max_safe_interval_days", plan.max_safe_interval_days],
  ];
  return (
    <div className="relational-plan-lab">
      <div className="natural-constraint-orb">
        <span>QUESTION</span>
        <i />
      </div>
      <div className="schema-router">
        <i />
        <strong>SCHEMA BOUNDARY</strong>
        <small>{asStrings(data.allowed_tables).length} allowed tables</small>
      </div>
      <div className="field-slots">
        {fields.map(([field, value], index) => (
          <div
            key={String(field)}
            style={customStyle({ "--delay": `${index * 120}ms` })}
          >
            <span>{String(field)}</span>
            <b>{text(value)}</b>
          </div>
        ))}
      </div>
    </div>
  );
}

function RowFilter({ data }: { data: JsonRecord }) {
  const filters = asRecords(data.filters);
  const max = Math.max(
    ...filters.map((item) => number(item.count)),
    1,
  );
  return (
    <div className="row-filter-tunnel">
      {filters.map((filter, index) => {
        const count = number(filter.count);
        const particles = Math.max(
          4,
          Math.min(48, Math.round((count / max) * 48)),
        );
        return (
          <div
            className="filter-gate"
            key={index}
            style={customStyle({
              "--ratio": count / max,
              "--delay": `${index * 180}ms`,
            })}
          >
            <header>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <strong>{text(filter.label)}</strong>
              <b>{count} rows</b>
            </header>
            <div className="row-particle-field">
              {Array.from({ length: particles }).map((_, particle) => (
                <i key={particle} style={customStyle({ "--index": particle })} />
              ))}
            </div>
            <code>
              {text(filter.field)} {text(filter.operator)}{" "}
              {text(filter.value)}
            </code>
          </div>
        );
      })}
    </div>
  );
}

function KeyJoin({ data }: { data: JsonRecord }) {
  const tables = asRecords(data.tables);
  const joins = asRecords(data.joins);
  const samples = asRecords(data.sample_rows);
  return (
    <div className="key-join-lab">
      <div className="table-space-3d">
        {tables.map((table, index) => (
          <div
            className="schema-table-plane"
            key={text(table.name)}
            style={customStyle({
              "--index": index,
              "--delay": `${index * 160}ms`,
            })}
          >
            <header>{text(table.name)}</header>
            {asStrings(table.columns).map((column) => (
              <span
                className={
                  column === text(table.primary_key) ||
                  text(table.primary_key).includes(column)
                    ? "key-column"
                    : ""
                }
                key={column}
              >
                <i />
                {column}
              </span>
            ))}
          </div>
        ))}
        <div className="join-beams">
          {joins.map((join, index) => (
            <div key={index} style={customStyle({ "--index": index })}>
              <i />
              <span>{text(join.left)}</span>
              <b>=</b>
              <span>{text(join.right)}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="joined-row-stream">
        <header>
          <span>JOINED ROW</span>
          <b>相等键值让分散字段进入同一逻辑行</b>
        </header>
        {samples.slice(0, 5).map((row, index) => (
          <div key={index} style={customStyle({ "--delay": `${index * 80}ms` })}>
            <span>{text(row.case_id)}</span>
            <span>{text(row.region_id)}</span>
            <span>{text(row.symptom_id)}</span>
            <strong>{text(row.disease_name)}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

function GroupAggregate({ data }: { data: JsonRecord }) {
  const diseases = asRecords(data.disease_counts);
  const total = number(data.matched_case_count);
  const max = Math.max(
    ...diseases.map((item) => number(item.case_count)),
    1,
  );
  return (
    <div className="group-aggregate-lab">
      <div className="group-source">
        <strong>{total}</strong>
        <span>joined rows</span>
        <div>
          {Array.from({ length: Math.min(total, 32) }).map((_, index) => (
            <i
              key={index}
              style={customStyle({
                "--index": index,
                "--orbit-radius": `${35 + (index % 5) * 9}px`,
              })}
            />
          ))}
        </div>
      </div>
      <div className="group-by-core">
        <i />
        <strong>GROUP BY</strong>
        <code>disease_id</code>
      </div>
      <div className="count-columns">
        {diseases.map((disease, index) => (
          <div
            key={text(disease.disease_id, String(index))}
            style={customStyle({
              "--ratio": number(disease.case_count) / max,
              "--delay": `${index * 180}ms`,
            })}
          >
            <i />
            <strong>{text(disease.disease_name)}</strong>
            <b>COUNT = {number(disease.case_count)}</b>
          </div>
        ))}
      </div>
    </div>
  );
}

function WorkspaceTableSchema({ data }: { data: JsonRecord }) {
  const columns = asRecords(data.columns);
  const indexes = new Set(asStrings(data.indexes));
  const preview = asRecords(data.preview);
  return (
    <div className="workspace-schema-space">
      <div className="schema-perspective-grid" aria-hidden="true" />
      <div className="schema-core-table">
        <header>
          <span>SQLITE TABLE</span>
          <strong>{text(data.table_name, "records")}</strong>
          <b>{number(data.row_count)} ROWS</b>
        </header>
        <div className="schema-column-stack">
          {columns.slice(0, 16).map((column, index) => (
            <div
              className={indexes.has(text(column.sql_name)) ? "indexed" : ""}
              key={text(column.sql_name, String(index))}
              style={customStyle({ "--delay": `${index * 65}ms` })}
            >
              <i />
              <strong>{text(column.source_name)}</strong>
              <code>{text(column.sql_name)}</code>
              <span>{text(column.type)}</span>
              {indexes.has(text(column.sql_name)) && <b>INDEX</b>}
            </div>
          ))}
        </div>
      </div>
      <div className="schema-sample-plane">
        <header><span>DATA PROFILE</span><b>{preview.length} sample rows</b></header>
        {preview.slice(0, 4).map((row, index) => (
          <div key={index} style={customStyle({ "--delay": `${index * 120}ms` })}>
            {Object.entries(row).slice(0, 4).map(([key, value]) => (
              <span key={key}><small>{key}</small><strong>{text(value)}</strong></span>
            ))}
          </div>
        ))}
      </div>
      <div className="schema-binding-beam">
        <i /><span>业务字段名</span><b>↔</b><span>安全 SQL 列名</span>
      </div>
    </div>
  );
}

function WorkspaceSqlPlan({ data }: { data: JsonRecord }) {
  const filters = asRecords(data.filters);
  const parameters = (data.parameters as JsonRecord | undefined) ?? {};
  const intent = text(data.intent, "select").toUpperCase();
  return (
    <div className="workspace-sql-planner">
      <div className="planner-intent-orb">
        <i /><i /><b>{intent}</b>
        <span>QUERY INTENT</span>
      </div>
      <div className="planner-routing-field">
        <header><span>{text(data.planner)}</span><b>SCHEMA BOUND</b></header>
        <div className="planner-route-beams">
          {filters.length ? filters.map((filter, index) => (
            <div key={index} style={customStyle({ "--delay": `${index * 150}ms` })}>
              <span>{text(filter.business_name, text(filter.column))}</span>
              <i />
              <code>{text(filter.column)} {text(filter.operator, "=")}</code>
              <b>{text(filter.value)}</b>
            </div>
          )) : (
            <div className="no-filter-route"><span>全表输入</span><i /><code>records</code><b>NO FILTER</b></div>
          )}
        </div>
      </div>
      <div className="parameter-vault">
        <span>PARAMETER VAULT</span>
        {Object.entries(parameters).length ? Object.entries(parameters).map(([key, value]) => (
          <div key={key}><code>:{key}</code><i /><strong>{text(value)}</strong></div>
        )) : <small>当前计划不需要外部参数</small>}
        <footer><i /> 值与 SQL 结构分离</footer>
      </div>
    </div>
  );
}

function WorkspaceTableResult({ data }: { data: JsonRecord }) {
  const columns = asStrings(data.columns);
  const rows = asRecords(data.rows);
  return (
    <div className="workspace-result-space">
      <div className="result-depth-axis"><span>SQLite</span><i /><span>RAG context</span></div>
      <div className="result-row-stack">
        {rows.slice(0, 10).map((row, index) => (
          <div
            key={index}
            style={customStyle({
              "--index": index,
              "--delay": `${index * 90}ms`,
              "--depth": `${Math.max(0, 72 - index * 8)}px`,
            })}
          >
            <header><span>ROW {String(index + 1).padStart(2, "0")}</span><i /></header>
            <section>
              {(columns.length ? columns : Object.keys(row)).slice(0, 6).map((column) => (
                <span key={column}><small>{column}</small><strong>{text(row[column])}</strong></span>
              ))}
            </section>
          </div>
        ))}
      </div>
      <div className="result-context-gate">
        <i /><b>{number(data.row_count)}</b><span>rows selected</span>
        <small>结构化行 → JSON 证据 → LLM</small>
      </div>
    </div>
  );
}

function SpatialRanking({ data }: { data: JsonRecord }) {
  const results = asRecords(data.results);
  const max = Math.max(
    ...results.map((item) => number(item.score)),
    0.000001,
  );
  return (
    <div className="spatial-ranking">
      <div className="ranking-depth-axis">
        <span>higher score</span>
        <i />
        <span>lower score</span>
      </div>
      <div className="ranking-cards-3d">
        {results.slice(0, 8).map((item, index) => (
          <div
            key={text(item.id, String(index))}
            style={customStyle({
              "--index": index,
              "--score": number(item.score) / max,
              "--delay": `${index * 120}ms`,
              "--card-left": `${2 + (index % 4) * 24}%`,
              "--card-top": `${Math.floor(index / 4) * 175}px`,
              "--card-depth": `${(1 - index / 8) * 75}px`,
              "--card-rotate": `${(index % 4 - 1.5) * 2}deg`,
            })}
          >
            <span>TOP {index + 1}</span>
            <strong>{text(item.title)}</strong>
            <p>{text(item.excerpt).slice(0, 86)}</p>
            <b>{number(item.score).toFixed(5)}</b>
          </div>
        ))}
      </div>
      <div className="evidence-exit">
        <i />
        <span>Top-K evidence → context window</span>
      </div>
    </div>
  );
}

export function DeepStageRenderer({
  stage,
}: {
  stage: TraceStage;
}): ReactElement | null {
  const data = stage.data as JsonRecord;
  switch (stage.kind) {
    case "tokens":
      return <TokenTrace data={data} />;
    case "vocabulary-space":
      return <VocabularySpace data={data} />;
    case "tfidf-build":
      return <TfidfBuild data={data} />;
    case "cosine-workbench":
      return <CosineWorkbench data={data} />;
    case "bm25-corpus":
      return <Bm25Corpus data={data} />;
    case "bm25-document":
      return <Bm25Document data={data} />;
    case "bm25-accumulator":
      return <Bm25Accumulator data={data} />;
    case "text":
      return <SemanticInput data={data} />;
    case "vector-index":
      return <VectorIndex data={data} />;
    case "dense-vector":
      return <DenseVector data={data} />;
    case "vector-space":
      return <VectorSpace data={data} />;
    case "dense-similarity":
      return <DenseSimilarity data={data} />;
    case "entity-space":
      return <EntitySpace data={data} />;
    case "graph-pattern":
      return <GraphPattern data={data} />;
    case "graph-traversal":
      return <GraphTraversal data={data} />;
    case "graph-aggregate":
      return <GraphAggregate data={data} />;
    case "ontology-space":
      return <OntologySpace data={data} />;
    case "iri-mapping":
      return <IriMapping data={data} />;
    case "triple-pattern":
      return <TriplePattern data={data} />;
    case "rdf-filter":
      return <RdfFilter data={data} />;
    case "relational-plan":
      return <RelationalPlan data={data} />;
    case "row-filter":
      return <RowFilter data={data} />;
    case "key-join":
      return <KeyJoin data={data} />;
    case "group-aggregate":
      return <GroupAggregate data={data} />;
    case "table-schema":
      return <WorkspaceTableSchema data={data} />;
    case "sql-plan":
      return <WorkspaceSqlPlan data={data} />;
    case "table-result":
      return <WorkspaceTableResult data={data} />;
    case "ranking":
      return <SpatialRanking data={data} />;
    default:
      return null;
  }
}
