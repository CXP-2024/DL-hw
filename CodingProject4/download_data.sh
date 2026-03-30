#!/bin/bash

set -euo pipefail

mkdir -p data

curl -C - -L -o data/icon-qa-train.arrow https://cloud.tsinghua.edu.cn/f/9607d2a4da3b4c338227/?dl=1
curl -C - -L -o data/icon-qa-val.arrow https://cloud.tsinghua.edu.cn/f/31b9920de29a41378e1a/?dl=1
