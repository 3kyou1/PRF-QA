

python train_roberta.py --data_path ../../../code/compress/results/stackexchange/qwen2/annotation_kept_cs512_stackexchange_zh_train_formated.pt \
    --save_path ../../../code/compress/results/models/xlm-roberta-large_stackexchange_only.pth

python model_set.py --model_name the base model path \
    --pth_path ../../../code/compress/results/models/bert-base_zh_stackexchange_only.pth \
    --save_path ../../../model/bert-base-stackexchange