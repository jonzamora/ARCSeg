python ../src/mainMiccaiSeg.py --save-dir=/data/miccaiSegResults --batchSize 1 --lr 0.0001 --epochs 2000 --saveTest True --resume save_MiccaiRecon/checkpoint_0.tar |& tee -a log_MiccaiSeg
