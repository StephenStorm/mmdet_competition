python tools/train.py configs/balloon/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_balloon.py


bash ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPU_NUM} \
    [optional arguments]

bash ./tools/dist_train.sh  configs/cascade_rcnn/tianchi_cascade_rcnn_r101_fphn_3x_ms.py 4