#!/bin/bash
# build.sh — 一键编译 LaTeX proposal
# 用法: ./build.sh
# 会执行 pdflatex + bibtex + pdflatex x2 的完整编译流程

set -e

export PATH="/Library/TeX/texbin:$PATH"

TEX_FILE="project_template"

echo "🔨 第 1 次 pdflatex..."
pdflatex -interaction=nonstopmode "$TEX_FILE.tex" > /dev/null 2>&1

echo "📚 bibtex..."
bibtex "$TEX_FILE" > /dev/null 2>&1 || true

echo "🔨 第 2 次 pdflatex..."
pdflatex -interaction=nonstopmode "$TEX_FILE.tex" > /dev/null 2>&1

echo "🔨 第 3 次 pdflatex..."
pdflatex -interaction=nonstopmode "$TEX_FILE.tex" > /dev/null 2>&1

echo ""
echo "✅ 编译完成！输出: ${TEX_FILE}.pdf"
echo "📄 页数: $(pdfinfo "$TEX_FILE.pdf" 2>/dev/null | grep Pages | awk '{print $2}' || echo '(需要安装 poppler 才能显示)')"
echo "💾 大小: $(du -h "$TEX_FILE.pdf" | awk '{print $1}')"
