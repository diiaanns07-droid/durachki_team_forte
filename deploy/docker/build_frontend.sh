#!/bin/sh
# Сборка frontend/ (A15) → $2. Нет frontend/ → страница-заглушка, сборка образа не падает.
set -eu
SRC="$1"; OUT="$2"; mkdir -p "$OUT"
if [ -f "$SRC/package.json" ]; then
  mkdir -p /build/fe && (cd "$SRC" && tar cf - --exclude=node_modules --exclude=dist .) | (cd /build/fe && tar xf -)
  cd /build/fe
  if [ -f package-lock.json ]; then npm ci; else npm install; fi
  npm run build
  for d in dist build out; do if [ -f "$d/index.html" ]; then cp -a "$d/." "$OUT/"; exit 0; fi; done
  echo "frontend: build без index.html" >&2; exit 1
elif [ -f "$SRC/index.html" ]; then
  cp -a "$SRC/." "$OUT/"
else
  printf '<!doctype html><meta charset="utf-8"><title>Kassa Sentinel</title><p>UI не собран. API: /docs</p>\n' > "$OUT/index.html"
fi
