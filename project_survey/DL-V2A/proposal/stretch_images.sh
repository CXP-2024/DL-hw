#!/bin/bash
# stretch_images.sh — 横向非等比拉伸图片（高度严格不变）
# 使用 macOS 原生 sips -z（零依赖）
# 用法: ./stretch_images.sh [拉伸比例] [图片1] [图片2] ...
# 示例: ./stretch_images.sh 1.3

set -e

SCALE="${1:-1.3}"
shift 2>/dev/null || true

if [ $# -eq 0 ]; then
  FILES=(fig_*.png)
else
  FILES=("$@")
fi

for img in "${FILES[@]}"; do
  if [ ! -f "$img" ]; then
    echo "⚠️  跳过：$img 不存在"
    continue
  fi

  W=$(sips -g pixelWidth "$img" | tail -1 | awk '{print $2}')
  H=$(sips -g pixelHeight "$img" | tail -1 | awk '{print $2}')
  NEW_W=$(python3 -c "import math; print(math.ceil($W * $SCALE))")

  echo "📐 $img: ${W}x${H} → ${NEW_W}x${H} (横向拉伸 ${SCALE}x)"

  # sips -z height width — 强制拉伸到精确尺寸，不保持比例
  sips -z "$H" "$NEW_W" "$img" --out "$img" >/dev/null 2>&1

  # 验证
  ACTUAL_W=$(sips -g pixelWidth "$img" | tail -1 | awk '{print $2}')
  ACTUAL_H=$(sips -g pixelHeight "$img" | tail -1 | awk '{print $2}')
  echo "✅ 已覆盖 $img (实际: ${ACTUAL_W}x${ACTUAL_H})"
done

echo ""
echo "✅ 全部完成！横向拉伸比例: ${SCALE}x"
