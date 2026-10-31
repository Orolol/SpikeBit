#!/usr/bin/env bash
# Build writeup/build/note.pdf with tectonic (XeTeX) and the arXiv source tarball
# writeup/build/arxiv.tar.gz (note.tex, note.bbl, the figure). arXiv compiles with pdflatex
# and does not run BibTeX, hence the .bbl and the ASCII-only check.
#
#     bash writeup/make_arxiv.sh
set -euo pipefail
cd "$(dirname "$0")"
if grep -nP '[^\x00-\x7F]' note.tex refs.bib; then
  echo "non-ASCII characters above: arXiv compiles with pdflatex" >&2
  exit 1
fi
mkdir -p build
tectonic -o build --keep-intermediates --keep-logs note.tex
if grep -nE 'undefined|Citation .* undefined|multiply defined' build/note.log; then
  echo "unresolved references above" >&2
  exit 1
fi
rm -rf build/arxiv
mkdir -p build/arxiv/figures
cp note.tex build/note.bbl build/arxiv/
cp figures/rho_star.pdf build/arxiv/figures/
tar czf build/arxiv.tar.gz -C build/arxiv .
tar tzf build/arxiv.tar.gz
echo "wrote writeup/build/note.pdf and writeup/build/arxiv.tar.gz"
