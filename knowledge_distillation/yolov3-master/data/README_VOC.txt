This folder contains helper files for using Pascal VOC with this repo.

- voc.names : class names (20 VOC classes).
- voc_v3.data : .data file pointing to train/val lists and names.
- train.txt / val.txt will be created in this folder by running:

  python convert_voc_to_yolo.py --images-root /path/to/voc/images \
    --annotations-root /path/to/voc/Annotations --out-data-dir data/voc

After running, start training with:

  python train.py --data data/voc_v3.data --cfg cfg/yolov3-spp.cfg --weights '' --batch-size 64 --device 7 --nw 32

 nohup python -u train.py --weights '' --data data/voc_v3.data --cfg cfg/yolov3-spp.cfg --batch-size 16 --device='4,5,6,7' --nw=48
python -u train.py --weights '' --data data/voc_syn.data --cfg cfg/yolov3-spp-voc.cfg --batch-size 16 --device=5 --nw=48  --resume
python -u train_v2.py --weights '/data1/home/ypliu/DIODE/knowledge_distillation/yolov3-master/runs/exp/weights/last.pt' --data data/voc_syn_512.data --cfg cfg/yolov3-spp-voc.cfg --batch-size 8 --device=0 --nw=48  --notest
python -u train_v2.py --weights '/data1/home/ypliu/DIODE/knowledge_distillation/yolov3-master/runs/exp/weights/last.pt' --data data/voc_random.data --cfg cfg/yolov3-spp-voc.cfg --batch-size 16 --device=0 --nw=48  --notest
