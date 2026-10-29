#!/usr/bin/env bash
#
# Download the DDMSC dump replayed by the NONREG test into
# tests/data/lofar_lba_point_sources/, replacing any existing copy.
#
# Usage:
#   ./scripts/fetch_test_data.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

TAG=nonreg-data-v1
ASSET=lofar_lba_point_sources.tar.gz
SHA256=140b2519fb3e1d196715c2b4abc8ff25ba420514af289622ef700d3f81ec94da

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

curl -fL --retry 3 -o "$TMP/$ASSET" "https://github.com/HugoGaquere/fast-deconv/releases/download/${TAG}/${ASSET}"
echo "${SHA256}  $TMP/$ASSET" | sha256sum -c -

mkdir -p tests/data
rm -rf tests/data/lofar_lba_point_sources
tar -xzf "$TMP/$ASSET" -C tests/data
