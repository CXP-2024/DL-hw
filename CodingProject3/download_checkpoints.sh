#!/bin/bash

set -euo pipefail

mkdir -p checkpoints

curl -C - -L -o checkpoints/MnistInceptionV3.pth https://cloud.tsinghua.edu.cn/f/257f2c04b2c741689abb/?dl=1
