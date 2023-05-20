#python main.py --epoch 500 --gap 100 --sparse 0.1
#python main.py --epoch 500 --gap 100 --sparse 0.3
CUDA_VISIBLE_DEVICES=0 python main.py --epoch 200 --gap 100 --sparse 0.1

CUDA_VISIBLE_DEVICES=0 python main.py --epoch 200 --gap 100 --sparse 0.3

CUDA_VISIBLE_DEVICES=0 python main.py --epoch 200 --gap 100 --sparse 0.5

CUDA_VISIBLE_DEVICES=0 python main.py --epoch 200 --gap 50 --sparse 0.1

CUDA_VISIBLE_DEVICES=0 python main.py --epoch 200 --gap 50 --sparse 0.3

CUDA_VISIBLE_DEVICES=0 python main.py --epoch 200 --gap 50 --sparse 0.5
#python main.py --epoch 500 --gap 50 --sparse 0.1