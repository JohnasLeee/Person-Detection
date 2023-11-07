from Detector import *

#modelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz'
#modelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz'
modelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet101_v1_1024x1024_coco17_tpu-8.tar.gz'
#modelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d7_coco17_tpu-32.tar.gz' #extremely slow
#modelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_512x512_coco17_tpu-8.tar.gz'



detector = Detector()
classFile = #PATH OF coco.name IN YOUR SYSTEM
imagePath = #PATH OF IMAGE IN YOUR SYSTEM
videoPath = #PATH OF VIDEO IN YOUR SYSTEM

threshold = 0.5
person = True
detector.readClasses(classFile)

detector.downloadModel(modelURL)

detector.loadModel()

detector.predictImage(imagePath,threshold)
#detector.predictVideo(videoPath,person,threshold)
