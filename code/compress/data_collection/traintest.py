import sys
import runpy
import os

os.chdir('/home/hang_su/per-agent/code/compress/data_collection')

args = 'python filter.py --load_path /home/hang_su/per-agent/code/compress/results/stackexchange/qwen2/annotation_cs512_stackexchange_train_formated.pt \
    --save_path /home/hang_su/per-agent/code/compress/results/stackexchange/qwen2/annotation_kept_cs512_stackexchange_train_formated.pt'

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


