#!/usr/bin/env bash
set -e
cd /root/hmp/myPLN
rm -f outputs/PLNResnet18/inject_attn_640/weights/async_eval/eval_4000_fix640.json
python examples/async_eval.py PLNResnet18 \
  --weights /root/hmp/myPLN/outputs/PLNResnet18/inject_attn_640/weights/weights_4000.pt \
  --output-json /root/hmp/myPLN/outputs/PLNResnet18/inject_attn_640/weights/async_eval/eval_4000_fix640.json \
  --batch 4000 \
  --batch-size 16 \
  --test-list /root/hmp/myPLN/data/voc_pln_cache/test_640.txt \
  --results-dir /root/hmp/myPLN/outputs/PLNResnet18/inject_attn_640/weights/async_eval/results_batch_4000_fix640 \
  --gpus 0,1 \
  > /root/hmp/myPLN/outputs/PLNResnet18/inject_attn_640/weights/async_eval/eval_4000_fix640.log 2>&1
