# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

python train_roberta.py --data_path /home/hang_su/per-agent/code/compress/results/stackexchange/qwen2/annotation_kept_cs512_stackexchange_zh_train_formated.pt \
    --save_path /home/hang_su/per-agent/code/compress/results/models/xlm-roberta-large_stackexchange_only.pth

python model_set.py --model_name /home/hang_su/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank \
    --pth_path /home/hang_su/per-agent/code/compress/results/models/bert-base_zh_stackexchange_only.pth \
    --save_path /home/hang_su/per-agent/model/bert-base-stackexchange