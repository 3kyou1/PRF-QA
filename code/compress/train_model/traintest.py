import sys
import runpy
import os

os.chdir('/home/hang_su/per-agent/code/compress/train_model')

args = 'python train_roberta.py --data_path /home/hang_su/per-agent/code/compress/results/stackexchange/qwen2/annotation_kept_cs512_stackexchange_train_formated.pt \
    --save_path /home/hang_su/per-agent/code/compress/results/models/xlm_roberta_large_stackexchange_only.pth'

args = args.split()
if args[0] == 'python':
    """pop up the first in the args""" 
    args.pop(0)

if args[0] == '-m':
    """pop up the first in the args"""
    args.pop(0)
    fun = runpy.run_module
else:
    fun = runpy.run_path

sys.argv.extend(args[1:])

fun(args[0], run_name='__main__')


