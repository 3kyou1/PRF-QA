python format_data.py

python compress.py --load_origin_from ../../../code/compress/results/stackexchange/origin/stackexchange.json  --chunk_size 512 --save_path ../../../code/compress/results/stackexchange/qwen2/compression_cs512_stackexchange_train_llmlingua2_zh_formated.json

python label_word.py  --load_prompt_from ../../../code/compress/results/stackexchange/qwen2/compression_cs512_stackexchange_zh_train_formated.json \
    --window_size 400 \
    --save_path ../../../code/compress/results/stackexchange/qwen2/annotation_cs512_stackexchange_zh_train_formated.json

python filter.py --load_path ../../../code/compress/results/stackexchange/qwen2/annotation_cs512_stackexchange_zh_train_formated.pt \
    --save_path ../../../code/compress/results/stackexchange/qwen2/annotation_kept_cs512_stackexchange_zh_train_formated.pt
