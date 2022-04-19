python tools/test.py \
    configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc.py \
    checkpoints/faster_rcnn_r50_fpn_1x_voc0712_20200624-c9895d40.pth \
    --eval mAP


./tools/dist_test.sh \
    configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
    checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
    8 \
    --out results.pkl \
    --eval bbox segm \
    --options "classwise=True"



python3 self_tools/inference.py --config configs/cascade_rcnn/tianchi_swinb_3x_bigsize_anchor_bs2x8_albu.py --checkpoint work_dirs/cascade_rcnn_swinb_fpn_3x_ms_albu_2lr/epoch_12.pth --out submit_0413_swinb_albu_2xlr.json

bash tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPU_NUM} \
    --format-only \
    --options ${JSONFILE_PREFIX} \
    [--show]

python tools/train.py \
    ${CONFIG_FILE} \
    [optional arguments]

    
./tools/dist_test.sh \
    configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
    checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth \
    8 \
    -format-only \
    --options "jsonfile_prefix=./mask_rcnn_test-dev_results"

./tools/dist_test.sh \
   /opt/tiger/haggs/mmdetection/work_dirs/cascade_rcnn_swinb_fpn_all_trick_sobel/tianchi_swinb_alltrick_sobel.py \
    /opt/tiger/haggs/mmdetection/work_dirs/cascade_rcnn_swinb_fpn_all_trick_sobel/epoch_12.pth \
    8 \
    --format-only \
    --options "jsonfile_prefix=/opt/tiger/haggs/mmdetection/work_dirs/cascade_rcnn_swinb_fpn_all_trick_sobel/res"
    

