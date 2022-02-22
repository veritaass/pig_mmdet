# MMDETECTION
## config 설정

https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html

## 사용모델 결정

성능평가 참고후 사용모델을 결정.

https://github.com/open-mmlab/mmdetection/blob/master/configs/yolox/README.md#results-and-models

configs/yolox/yolox_x_8x8_300e_coco.py

## 설정 변경

```
_base_ = '../yolox/yolox_s_8x8_300e_coco.py'

# model settings
# pretrained = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'
pretrained = '/home/bkseo_ext/git/mmdet220/mmdetection/work_dirs/bk_yolox_onlybbox/pre-4800.pth'
model = dict(
    backbone=dict(deepen_factor=1.33, widen_factor=1.25,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        in_channels=[320, 640, 1280], out_channels=320, num_csp_blocks=4),
    bbox_head=dict(in_channels=320, feat_channels=320))

# customizations
# data_root = '/raid/templates/data/pig-bbox/'
data_root = '/raid/templates/data/pig-bbox/single/'
#classes = ('pig','boar','sow', 'piglet', 'baby-pig')
classes = ('pig',)
dataset_type = 'CocoDataset'
model = dict(
    bbox_head=dict(
        type='YOLOXHead', num_classes=1, in_channels=128, feat_channels=128),)


train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_Train.json',
        img_prefix=data_root + 'images/',
        classes=classes,
    ),)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_Validation.json',
        img_prefix=data_root + 'images/',
        classes=classes,
        ),
    test=dict(
        type=dataset_type,
         ann_file=data_root + 'annotations/instances_Test.json',
        img_prefix=data_root + 'images/',
        classes=classes
        ))

runner = dict(type='EpochBasedRunner', max_epochs=9600)
```

## 학습
```
python tools/train.py 
usage: train.py [-h] [--work-dir WORK_DIR] [--resume-from RESUME_FROM]  
                [--no-validate]  
                [--gpus GPUS | --gpu-ids GPU_IDS [GPU_IDS ...]] [--seed SEED]  
                [--deterministic] [--options OPTIONS [OPTIONS ...]]  
                [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]]  
                [--launcher {none,pytorch,slurm,mpi}]  
                [--local_rank LOCAL_RANK]  
                config  
train.py: error: the following arguments are required: config
```
=> python tools/train.py configs/pig/bk_yolox_onlybbox.py

## 최종 설정

최종설정사항은 다음의 화일에서 확인가능 
work_dir/{config}/{config}.py 

  

# 데이터수집

데이터 수집도구 

youtube 다운로드

```
import argparse
import pytube # pip install pytube

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--url", required=True, help="youtube url")
args = vars(ap.parse_args())

streams = pytube.YouTube(args["url"]).streams.all()
for stream in streams:
    print(stream)

print('press number key to select video index')
s = input()
streams[int(s)].download()
```

## 이미지 다운로드 (1/2)  

chrome => developer mode => console, paste and run.

```
/**
 * simulate a right-click event so we can grab the image URL using the
 * context menu alleviating the need to navigate to another page
 *
 * attributed to @jmiserez: http://pyimg.co/9qe7y
 *
 * @param   {object}  element  DOM Element
 *
 * @return  {void}
 */
function simulateRightClick( element ) {
    var event1 = new MouseEvent( 'mousedown', {
        bubbles: true,
        cancelable: false,
        view: window,
        button: 2,
        buttons: 2,
        clientX: element.getBoundingClientRect().x,
        clientY: element.getBoundingClientRect().y
    } );
    element.dispatchEvent( event1 );
    var event2 = new MouseEvent( 'mouseup', {
        bubbles: true,
        cancelable: false,
        view: window,
        button: 2,
        buttons: 0,
        clientX: element.getBoundingClientRect().x,
        clientY: element.getBoundingClientRect().y
    } );
    element.dispatchEvent( event2 );
    var event3 = new MouseEvent( 'contextmenu', {
        bubbles: true,
        cancelable: false,
        view: window,
        button: 2,
        buttons: 0,
        clientX: element.getBoundingClientRect().x,
        clientY: element.getBoundingClientRect().y
    } );
    element.dispatchEvent( event3 );
}

/**
 * grabs a URL Parameter from a query string because Google Images
 * stores the full image URL in a query parameter
 *
 * @param   {string}  queryString  The Query String
 * @param   {string}  key          The key to grab a value for
 *
 * @return  {string}               value
 */
function getURLParam( queryString, key ) {
    var vars = queryString.replace( /^\?/, '' ).split( '&' );
    for ( let i = 0; i < vars.length; i++ ) {
        let pair = vars[ i ].split( '=' );
        if ( pair[0] == key ) {
            return pair[1];
        }
    }
    return false;
}

/**
 * Generate and automatically download a txt file from the URL contents
 *
 * @param   {string}  contents  The contents to download
 *
 * @return  {void}
 */
function createDownload( contents ) {
    var hiddenElement = document.createElement( 'a' );
    hiddenElement.href = 'data:attachment/text,' + encodeURI( contents );
    hiddenElement.target = '_blank';
    hiddenElement.download = 'urls.txt';
    hiddenElement.click();
}

/**
 * grab all URLs va a Promise that resolves once all URLs have been
 * acquired
 *
 * @return  {object}  Promise object
 */
function grabUrls() {
    var urls = [];
    return new Promise( function( resolve, reject ) {
        var count = document.querySelectorAll(
        	'.isv-r a:first-of-type' ).length,
            index = 0;
        Array.prototype.forEach.call( document.querySelectorAll(
        	'.isv-r a:first-of-type' ), function( element ) {
            // using the right click menu Google will generate the
            // full-size URL; won't work in Internet Explorer
            // (http://pyimg.co/byukr)
            simulateRightClick( element.querySelector( ':scope img' ) );
            // Wait for it to appear on the <a> element
            var interval = setInterval( function() {
                if ( element.href.trim() !== '' ) {
                    clearInterval( interval );
                    // extract the full-size version of the image
                    let googleUrl = element.href.replace( /.*(\?)/, '$1' ),
                        fullImageUrl = decodeURIComponent(
                        	getURLParam( googleUrl, 'imgurl' ) );
                    if ( fullImageUrl !== 'false' ) {
                        urls.push( fullImageUrl );
                    }
                    // sometimes the URL returns a "false" string and
                    // we still want to count those so our Promise
                    // resolves
                    index++;
                    if ( index == ( count - 1 ) ) {
                        resolve( urls );
                    }
                }
            }, 10 );
        } );
    } );
}

/**
 * Call the main function to grab the URLs and initiate the download
 */
grabUrls().then( function( urls ) {
    urls = urls.join( '\n' );
    createDownload( urls );
} );
```

## 이미지 다운로드 (2/2)
```
# import the necessary packages
from imutils import paths
import argparse
import requests
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--urls", required=True,
	help="path to file containing image URLs")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory of images")
args = vars(ap.parse_args())
# grab the list of URLs from the input file, then initialize the
# total number of images downloaded thus far
rows = open(args["urls"]).read().strip().split("\n")
total = 0

# loop the URLs
for url in rows:
	try:
		# try to download the image
		r = requests.get(url, timeout=60)
		# save the image to disk
		p = os.path.sep.join([args["output"], "{}.jpg".format(
			str(total).zfill(8))])
		f = open(p, "wb")
		f.write(r.content)
		f.close()
		# update the counter
		print("[INFO] downloaded: {}".format(p))
		total += 1
	# handle if any exceptions are thrown during the download process
	except:
		print("[INFO] error downloading {}...skipping".format(p))

# loop over the image paths we just downloaded
for imagePath in paths.list_images(args["output"]):
	# initialize if the image should be deleted or not
	delete = False
	# try to load the image
	try:
		image = cv2.imread(imagePath)
		# if the image is `None` then we could not properly load it
		# from disk, so delete it
		if image is None:
			delete = True
	# if OpenCV cannot load the image then the image is likely
	# corrupt so we should delete it
	except:
		print("Except")
		delete = True
	# check to see if the image should be deleted
	if delete:
		print("[INFO] deleting {}".format(imagePath))
		os.remove(imagePath)


```

## 중복이미지 제거

```
# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import cv2
import os


def dhash(image, hashSize=8):
    # convert the image to grayscale and resize the grayscale image,
    # adding a single column (width) so we can compute the horizontal
    # gradient
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))

    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]

    # convert the difference image to a hash and return it
    return sum([2**i for (i, v) in enumerate(diff.flatten()) if v])


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument(
    "-r",
    "--remove",
    type=int,
    default=-1,
    help="whether or not duplicates should be removed (i.e., dry run)")
args = vars(ap.parse_args())

# grab the paths to all images in our input dataset directory and
# then initialize our hashes dictionary
print("[INFO] computing image hashes...")
imagePaths = list(paths.list_images(args["dataset"]))
hashes = {}

# loop over our image paths
for imagePath in imagePaths:
    # load the input image and compute the hash
    image = cv2.imread(imagePath)
    h = dhash(image)

    # grab all image paths with that hash, add the current image
    # path to it, and store the list back in the hashes dictionary
    p = hashes.get(h, [])
    p.append(imagePath)
    hashes[h] = p

# loop over the image hashes
for (h, hashedPaths) in hashes.items():
    # check to see if there is more than one image with the same hash
    if len(hashedPaths) > 1:
        # check to see if this is a dry run
        if args["remove"] <= 0:
            # initialize a montage to store all images with the same
            # hash
            montage = None

            # loop over all image paths with the same hash
            for p in hashedPaths:
                # load the input image and resize it to a fixed width
                # and heightG
                image = cv2.imread(p)
                image = cv2.resize(image, (150, 150))

                # if our montage is None, initialize it
                if montage is None:
                    montage = image

                # otherwise, horizontally stack the images
                else:
                    montage = np.hstack([montage, image])

            # show the montage for the hash
            print("[INFO] hash: {}".format(h))
            cv2.imshow("Montage", montage)
            cv2.waitKey(0)

# otherwise, we'll be removing the duplicate images
        else:
            # loop over all image paths with the same hash *except*
            # for the first image in the list (since we want to keep
            # one, and only one, of the duplicate images)
            for p in hashedPaths[1:]:
                os.remove(p)
```
