# support keypoint as cvat.


import time
import logging
from typing import Dict
import mmcv
import os
import cv2
import pathlib
import json
import pathlib
import numpy as np
from pycocotools import coco
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from argparse import ArgumentParser
from mmseg.apis import inference_segmentor, init_segmentor
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
from mmdet.apis import inference_detector, init_detector
from pycocotools.coco import COCO
from xml.dom import minidom

# model
det_model = None
pose_model = None
seg_model = None




class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class AnnoEventHandler(FileSystemEventHandler):
    """Logs all the events captured."""

    def __init__(self, logger=None):
        super().__init__()

        self.logger = logger or logging.root
        self.last_modified = None
        self.last_size = -1

    def on_moved(self, event):
        super().on_moved(event)

        what = 'directory' if event.is_directory else 'file'
        self.logger.info("Moved %s: from %s to %s", what, event.src_path,
                         event.dest_path)

        if what is 'directory':
            return
        if pathlib.Path(event.dest_path).suffix != '.mp4':
            return
        if event.dest_path == event.src_path:
            return

        # only .mp4 file
        self.logger.info("mp4 detected moved %s: %s",
                         what, event.dest_path)

        process_mp4(event.dest_path)



    def on_created(self, event):
        super().on_created(event)

        what = 'directory' if event.is_directory == True else 'file'
        self.logger.info("Created %s: %s", what, event.src_path)

    def on_deleted(self, event):
        super().on_deleted(event)

        what = 'directory' if event.is_directory == True else 'file'
        self.logger.info("Deleted %s: %s", what, event.src_path)

    def on_modified(self, event):
        super().on_modified(event)
        what = 'directory' if event.is_directory else 'file'

        if what is 'directory':
            return
        if pathlib.Path(event.src_path).suffix != '.mp4':
            return
        if self.last_modified == event.src_path and self.last_size == os.path.getsize(event.src_path):
            return

        # only .mp4 file
        self.logger.info("mp4 detected modified %s: %s",
                         what, event.src_path)
        size_past = os.path.getsize(event.src_path)

        while True:
            time.sleep(1)
            size_now = os.path.getsize(event.src_path)
            if size_now == size_past:
                self.logger.debug(
                    "file has copied completely now size: %s", size_now)
                self.last_modified = event.src_path
                self.last_size = size_now
                process_mp4(event.src_path)

                break
                # TODO: why sleep is not working here ?
            else:
                size_past = os.path.getsize(event.src_path)
                self.logger.debug("file copying size: %s", size_past)


def process_mmdet_results(mmdet_results, cat_id=1):
    """Process mmdet results, and return a list of bboxes.

    :param mmdet_results:
    :param cat_id: category id (default: 1 for human)
    :return: a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results

    bboxes = det_results[cat_id - 1]

    person_results = []
    for bbox in bboxes:
        person = {}
        person['bbox'] = bbox
        person_results.append(person)

    return person_results


def create_image(seq, width, height, file_name):
    ann_image = {
        "id": seq,
        "width": width,
        "height": height,
        "file_name": file_name,
    }
    return ann_image


def create_coco_anno_obj():
    coco = dict()
    coco['images'] = []

    coco['categories'] = [
        {
            "id": 1,
            "name": "Cow",
            "supercategory": "Animals",
            "color": "#1bbe19",
            "metadata": {},
            "keypoint_colors": []
        }
    ]
    coco['annotations'] = []
    return coco

def create_annotation():
    return {
        "id": -1,
        "image_id": -1,
        "category_id": 1,
        "segmentation": [
            [
                769.0,
                238.0,
                769.0,
                632.0,
                1307.0,
                632.0,
                1307.0,
                238.0
            ]
        ],
        "area": 211972,
        "bbox": [
            769.0,
            238.0,
            538.0,
            394.0
        ],
        "iscrowd": False,
        "isbbox": True,
        "color": "#1bd838",
        "keypoints": [
            831,
            464,
            1,
            1246,
            538,
            1
        ],
        "num_keypoints": 20
    }
def append_bbox_annotation(coco_obj, image_id, bboxes):
    for bbox in bboxes:
        annotation = create_annotation()
        annotation['id'] = len(coco_obj["annotations"])
        annotation['image_id'] = image_id
        annotation['segmentation'] = []
        ibox = []
        ibox.append(int(bbox["bbox"][0]))
        ibox.append(int(bbox["bbox"][1]))
        ibox.append(int(bbox["bbox"][2]) - int(bbox["bbox"][0]))
        ibox.append(int(bbox["bbox"][3]) - int(bbox["bbox"][1]))
        annotation["bbox"] = ibox
        annotation["area"] = ibox[2] * ibox[3]
        coco_obj["annotations"].append(annotation)
        del annotation['keypoints']
        del annotation['num_keypoints']

    return coco_obj

def append_keyp_annotation(coco_obj, image_id, pose_results):
    for pose in pose_results:
        annotation = create_annotation()
        annotation['id'] = len(coco_obj["annotations"])
        annotation['image_id'] = image_id
        annotation['segmentation'] = []

        keypoints = []
        for keyp in pose['keypoints']:
            keypoints.append(int(keyp[0]))
            keypoints.append(int(keyp[1]))
            keypoints.append(1)
        annotation['keypoints'] = keypoints
        annotation['num_keypoints'] = 20
        del annotation['bbox']
        del annotation['area']
        annotation["isbbox"] = False
        coco_obj["annotations"].append(annotation)
        

    return coco_obj


# path => location/infer-type/filename.mp4
def process_mp4(fpath):
    logger = logging.root
    BBOX = 'Bounding-box'
    KEYP = 'Key-point'
    POLY = 'Polygon'

    paths = fpath.split('/')
    fname = paths[-1]
    inf_type = paths[-2]
    location = paths[-3]

    if inf_type not in [BBOX, KEYP, POLY]:
        logger.info("Not a type to process. ")
        return

    global det_model, pose_model, seg_model


    video = mmcv.VideoReader(fpath)
    logger.info(f'fps: {video._fps}, frame_cnt: {video._frame_cnt}')
    

    paths[-4] = 'send'
    send_path = '/'.join(paths)
    pathlib.Path(send_path).mkdir(parents=True, exist_ok=True)

    tot_frames = 0
    coco_obj = create_coco_anno_obj()
    
    for frame in video:
        tot_frames += 1
        # if tot_frames == 50:
        #     break
        frame_name = f'frame-{tot_frames:05d}.jpg'
        if tot_frames % (video._fps / 2) == 0:
            cv2.imwrite(os.path.join(send_path, frame_name), frame)
            ann_image = create_image(
                seq=tot_frames, width=video._width, height=video._height, file_name=frame_name)
            coco_obj['images'].append(ann_image)


            if inf_type == POLY:
                seg_result = inference_segmentor(seg_model, frame)
                fr_resfile = os.path.join(
                    send_path, f"seg-{tot_frames:05d}.json")
                _ = seg_model.show_result(frame, seg_result, opacity=0.1, out_file=os.path.join(
                    send_path, f"seg-{tot_frames:05d}.jpg"))
                with open(fr_resfile, 'w', encoding='utf8') as json_file:
                    json.dump(seg_result, json_file,
                              cls=NumpyEncoder, ensure_ascii=False)
            else:

                # BBOX
                mmdet_results = inference_detector(det_model, frame)
                det_results = process_mmdet_results(
                    mmdet_results, 20)  # 20 for cow

                # logger.info(f'length of detection results : {len(det_results)}, {det_results}')
                if inf_type == BBOX:
                    coco_obj = append_bbox_annotation(coco_obj, tot_frames, det_results)

                elif inf_type == KEYP:
                    pose_results, returned_outputs = inference_top_down_pose_model(
                        pose_model,
                        frame,
                        det_results,
                        bbox_thr=0.4,
                        format='xyxy',
                        dataset='AnimalPoseDataset',  # bkseo temp.,
                        return_heatmap=None,
                        outputs=None)

                    vis_pose_result(
                        pose_model,
                        frame,
                        pose_results,
                        dataset='AnimalPoseDataset',
                        kpt_score_thr=0.3,
                        radius=1,
                        thickness=1,
                        show=False,
                        out_file=os.path.join(send_path, f'pose-{tot_frames:05d}.jpg'))

                    coco_obj = append_keyp_annotation(coco_obj, tot_frames, pose_results=pose_results)
                    


    with open(os.path.join(send_path, f"coco-annotation.json"), 'w', encoding='utf8') as json_file:
        json.dump(coco_obj, json_file, cls=NumpyEncoder, ensure_ascii=False)

    logger.info(f"Done JSON : {send_path}")


    # now writing CVA xml
    logger.info(f"Creating XML.")
    create_cvat_xml(coco_obj, send_path)
    logger.info(f"Done XML.")


def create_cvat_xml(cocoset, send_path):

    xml_doc = minidom.Document()
    xml_root = xml_doc.createElement('annotations') 
    xml_doc.appendChild(xml_root)

    lst_img = cocoset['images']

    for i in range(0,len(lst_img)):
        img_id = str(lst_img[i]['id'])
        img_name = str(lst_img[i]['file_name'])    
        xml_image = xml_doc.createElement('image')
        xml_image.setAttribute('id', img_id)
        xml_image.setAttribute('name', img_name)
        xml_root.appendChild(xml_image)

    xml_output = xml_root.toprettyxml(indent ="\t") 
    lst_anno = cocoset["annotations"]
    label="Cow"
    occluded="0"

    num_keypoint = 20
    for anno in range(1,len(lst_anno)):
        image_id = str(lst_anno[anno]['image_id'])
        points = lst_anno[anno]['keypoints']
        
        el_image = xml_root.getElementsByTagName('image')
        
        for el in el_image:
            if el.getAttribute('id') == image_id:            
                str_points = ""
                for p in range(0,60,3):
                    str_points += str(points[p])+","+str(points[p+1])+";"            
                xml_points = xml_doc.createElement('points')
                xml_points.setAttribute('label', label)
                xml_points.setAttribute('occluded', occluded)
                xml_points.setAttribute('points', str_points[0:-1])
                el.appendChild(xml_points)            
                xml_output = xml_root.toprettyxml(indent ="\t") 
                
    # with open(os.path.join(send_path, f"cvat-keypoint.xml"), 'w', encoding='utf8') as f:
    with open(os.path.join(send_path, f"cvat-keypoint.xml"), 'w') as f:
        f.write(xml_root.toxml())

    pass

# def get_model_files(zoo_id):
#     return zoo[zoo_id]['config'], zoo[zoo_id]['checkpoint']

def main():
    global det_model, pose_model, seg_model

    parser = ArgumentParser()
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--out-video-root', type=str, default='.', help='Output directory')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')

    parser.add_argument(
        '--fps',
        type=int,
        default=2,
        help='fps. should be int.')
    parser.add_argument(
        'recv_dir',
        help='Specify directory to monitor')

    parser.add_argument('--det-config',
                        default='../mmdetection/configs/mask_rcnn/mask_rcnn_r101_fpn_mstrain-poly_3x_coco.py',
                        help='Config file for detection')
    parser.add_argument('--det-checkpoint',
                        default='https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r101_fpn_mstrain-poly_3x_coco/mask_rcnn_r101_fpn_mstrain-poly_3x_coco_20210524_200244-5675c317.pth',
                        help='Checkpoint file for detection')

    parser.add_argument('--pose_config',
                        default='../mmpose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/hrnet_w48_animalpose_256x256.py',
                        help='Config file for pose')
    parser.add_argument('--pose_checkpoint',
                        default='https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w48_animalpose_256x256-34644726_20210426.pth',
                        help='Checkpoint file for pose')

    parser.add_argument('--seg_config', default='../mmsegmentation/configs/pspnet/pspnet_r101-d8_512x512_40k_voc12aug.py',
                        help='Config file for segmentation')
    parser.add_argument('--seg_checkpoint',
                        default='../mmsegmentation/checkpoints/pspnet_r101-d8_512x512_40k_voc12aug_20200613_161222-bc933b18.pth',
                        help='Checkpoint file for segmentation')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.root
    path = args.recv_dir
    logger.info("Detection initlizing ...")
    det_model = init_detector(
        args.det_config, args.det_checkpoint, device='cuda:5')
    logger.info("PoseModel initlizing ...")
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device='cuda:6')
    logger.info("Segmentation initlizing ...")
    seg_model = init_segmentor(
        args.seg_config, args.seg_checkpoint, device='cuda:7')
    event_handler = AnnoEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
