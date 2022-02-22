nohup python main3.py /raid/templates/farm-data/refine_data/receive/ > processed-cow.log 2>&1 &
nohup python main3.py /raid/templates/farm-data/refine_data/pig/receive/ --det-config=../mmdet220/mmdetection/configs/pig/bk_yolox_onlybbox.py --det-checkpoint=../mmdet220/mmdetection/work_dirs/bk_yolox_onlybbox/latest-5990.pth > processed-pig.log 2>&1 &
