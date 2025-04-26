#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --epochs) # e.g. 50
      EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    --model-size) # e.g. one of: s, m, l
      MODEL_SIZE="$2"
      shift # past argument
      shift # past value
      ;;
    --batch-size) # e.g. 16
      BATCH_SIZE="$2"
      shift # past argument
      shift # past value
      ;;
    --data-directory) # e.g. 0.2
      DATA_DIRECTORY="$2"
      shift # past argument
      shift # past value
      ;;
    --out-directory) # e.g. (96,96,3)
      OUT_DIRECTORY="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [ -z "$EPOCHS" ]; then
    echo "Missing --epochs"
    exit 1
fi
if [ -z "$MODEL_SIZE" ]; then
    echo "Missing --model-size"
    exit 1
fi
if [[ "$MODEL_SIZE" != "s" && "$MODEL_SIZE" != "m" && "$MODEL_SIZE" != "l" ]]; then
    echo "Invalid --model-size '"$MODEL_SIZE"', expected 's', 'm' or 'l'"
    exit 1
fi
if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=16
fi
if [ -z "$DATA_DIRECTORY" ]; then
    echo "Missing --data-directory"
    exit 1
fi
if [ -z "$OUT_DIRECTORY" ]; then
    echo "Missing --out-directory"
    exit 1
fi

OUT_DIRECTORY=$(realpath $OUT_DIRECTORY)
DATA_DIRECTORY=$(realpath $DATA_DIRECTORY)

IMAGE_SIZE=$(python3 get_image_size.py --data-directory "$DATA_DIRECTORY")

# surpress OpenBLAS warnings
export OMP_NUM_THREADS=1

# convert Edge Impulse dataset (in Numpy format, with JSON for labels into something YOLOv5 understands)
python3 -u extract_dataset.py --data-directory $DATA_DIRECTORY --out-directory /tmp/data

cd /app/yolov5
rm -rf ./runs/train/yolov5_results/

# train:
#     --freeze 10 - freeze the bottom layers of the network
#     --workers 0 - as this otherwise requires a larger /dev/shm than we have on Edge Impulse prod,
#                   there's probably a workaround for this, but we need to check with infra.
python3 -u train.py --img $IMAGE_SIZE \
    --freeze 10 \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --data /tmp/data/data.yaml \
    --weights /app/yolov5$MODEL_SIZE.pt \
    --name yolov5_results \
    --cache \
    --workers 0
echo "Training complete"
echo ""

mkdir -p $OUT_DIRECTORY

# export as onnx
echo "Converting to ONNX..."
python3 -u models/export.py  --weights ./runs/train/yolov5_results/weights/last.pt --img-size $IMAGE_SIZE --batch-size 1 --grid
cp ./runs/train/yolov5_results/weights/last.onnx $OUT_DIRECTORY/model.onnx
# add shape info
python3 /scripts/add_shape_info.py --onnx-file $OUT_DIRECTORY/model.onnx
echo "Converting to ONNX OK"
echo ""

# export as f32
echo "Converting to TensorFlow Lite model (fp16)..."
cp $OUT_DIRECTORY/model.onnx /tmp/model.onnx
# strip off all but one output layers
if [ "$MODEL_SIZE" == "s" ]; then
  snd4onnx -rn 397 672 947 -if /tmp/model.onnx -of /tmp/model.onnx --non_verbose
elif [ "$MODEL_SIZE" == "m" ]; then
  snd4onnx -rn 524 799 1074 -if /tmp/model.onnx -of /tmp/model.onnx --non_verbose
elif [ "$MODEL_SIZE" == "l" ]; then
  snd4onnx -rn 651 926 1201 -if /tmp/model.onnx -of /tmp/model.onnx --non_verbose
fi
# Convert to NHWC
python3 /scripts/convert-to-nhwc.py --onnx-file /tmp/model.onnx --out-file /tmp/model.onnx
# Convert to TFLite
python3 -u /scripts/onnx_to_tflite.py --onnx-file /tmp/model.onnx --out-file $OUT_DIRECTORY/model.tflite
echo "Converting to TensorFlow Lite model (fp16) OK"
echo ""
