#!/usr/bin/env bash
# 兼容入口：真正的脚本已收进 fit_scripts/（见 fit_scripts/README.md）。
# 保留这个转发是为了不破坏既有用法：bash /path/to/pheasy-gpu/fit_3090.sh ...
set -euo pipefail
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$_here/fit_scripts/fit_3090.sh" "$@"
