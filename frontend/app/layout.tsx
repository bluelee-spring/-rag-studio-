import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG 方法实验台",
  description: "多范式检索增强生成教学系统",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}

