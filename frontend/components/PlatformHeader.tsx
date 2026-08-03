"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

const NAVIGATION = [
  { href: "/", label: "实验台" },
  { href: "/data", label: "数据工作区" },
  { href: "/batch", label: "批量分类" },
  { href: "/settings", label: "模型连接" },
];

export function PlatformHeader({
  serviceOnline,
  modelReady,
}: {
  serviceOnline?: boolean;
  modelReady?: boolean;
}) {
  const pathname = usePathname();
  const [pendingHref, setPendingHref] = useState("");

  useEffect(() => {
    setPendingHref("");
  }, [pathname]);

  return (
    <header className="topbar platform-topbar">
      <Link className="brand" href="/" aria-label="RAG平台首页">
        <span className="brand-mark">R</span>
        <span>
          <strong>RAG STUDIO</strong>
          <small>Retrieval Engineering</small>
        </span>
      </Link>
      <nav className="platform-nav" aria-label="平台导航">
        {NAVIGATION.map((item) => {
          const active =
            item.href === "/"
              ? pathname === "/"
              : pathname === item.href || pathname.startsWith(`${item.href}/`);
          const pending = pendingHref === item.href;
          return (
            <Link
              aria-current={active ? "page" : undefined}
              className={`${active ? "active" : ""} ${pending ? "pending" : ""}`.trim()}
              href={item.href}
              key={item.href}
              onClick={() => {
                if (!active) setPendingHref(item.href);
              }}
              prefetch={false}
            >
              <i />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>
      <div className="topbar-status">
        {typeof serviceOnline === "boolean" && (
          <>
            <span className={serviceOnline ? "status-dot online" : "status-dot"} />
            {serviceOnline ? "数据服务已连接" : "等待后端服务"}
          </>
        )}
        {typeof modelReady === "boolean" && (
          <span className={modelReady ? "model-status connected" : "model-status"}>
            {modelReady ? "生成模型可用" : "生成模型未连接"}
          </span>
        )}
      </div>
    </header>
  );
}
