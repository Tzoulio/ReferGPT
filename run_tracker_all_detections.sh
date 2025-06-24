#!/bin/bash

# === CONFIGURATION FILES ===
CAR_CONFIG_FILE="config/global/cfg_refergpt.yaml"
PEDESTRIAN_CONFIG_FILE="config/global/cfg_refergpt_pedestrian.yaml"
PYTHON_SCRIPT="main.py"
EVAL_SCRIPT="evaluate_refergpt.sh"

# === DETECTION & LLM PATHS ===
DETECTIONS_PATHS=(
  "data/detections/pvrcnn/training"
  "data/detections/qt-3dt/training"
  "data/detections_casa"
  "data/detections/point_rcnn/training"
  "data/detections/virconv/training"
  "data/detections/second_iou/training"
)

LLM_OUTPUT_FILES=(
  "dataset/data/updated_pvrcnn_llm_output_data"
  "dataset/data/updated_qt_3dt_llm_output_data"
  "dataset/data/updated_casa_llm_output_data"
  "dataset/data/updated_point_rcnn_llm_output_data"
  "dataset/data/updated_virconv_llm_output_data"
  "dataset/data/updated_second_iou_llm_output_data"
)

LLM_OUTPUT_FILES_PEDESTRIAN=(
  "dataset/data/updated_pvrcnn_llm_output_data_pedestrian"
  "dataset/data/updated_qt_3dt_llm_output_data_pedestrian"
  "dataset/data/updated_casa_llm_output_data_pedestrian"
  "dataset/data/updated_casa_llm_output_data_pedestrian"
  "dataset/data/updated_casa_llm_output_data_pedestrian"
  "dataset/data/updated_second_iou_llm_output_data_pedestrian"
)

RESULTS_OUTPUT_TEXT=(
  "/home/leandro/Documents/TrackGPT/refergpt/evaluation/results/sha_key/data/refer-kitti/output/pvrcnn.txt"
  "/home/leandro/Documents/TrackGPT/refergpt/evaluation/results/sha_key/data/refer-kitti/output/qt_3dt.txt"
  "/home/leandro/Documents/TrackGPT/refergpt/evaluation/results/sha_key/data/refer-kitti/output/casa.txt"
  "/home/leandro/Documents/TrackGPT/refergpt/evaluation/results/sha_key/data/refer-kitti/output/point_rcnn.txt"
  "/home/leandro/Documents/TrackGPT/refergpt/evaluation/results/sha_key/data/refer-kitti/output/virconv.txt"
  "/home/leandro/Documents/TrackGPT/refergpt/evaluation/results/sha_key/data/refer-kitti/output/second_iou.txt"
)

post_score_car=(
  1.6
  0.2
  1.6
  1.6
  1.6
  0.0
)

post_score_pedestrian=(
  3.0
  0.4
  4.0
  4.0
  4.0
  0.3
)

# === FUNCTION TO UPDATE CONFIG AND RUN ===
run_with_tracking_type() {
  local tracking_type="$1"       # Car or Pedestrian
  local distance_threshold="$2"  # 150 or 180
  local index="$3"               # Current iteration index
  
  # Select appropriate config file and params
  if [ "$tracking_type" = "Car" ]; then
    CONFIG_FILE="$CAR_CONFIG_FILE"
    POST_SCORE="${post_score_car[$index]}"
    LLM_OUTPUT_FILE="${LLM_OUTPUT_FILES[$index]}"
  elif [ "$tracking_type" = "Pedestrian" ]; then
    CONFIG_FILE="$PEDESTRIAN_CONFIG_FILE"
    POST_SCORE="${post_score_pedestrian[$index]}"
    LLM_OUTPUT_FILE="${LLM_OUTPUT_FILES_PEDESTRIAN[$index]}"
  else
    echo "Unknown tracking type: $tracking_type"
    exit 1
  fi
  
  echo "=== Running for $tracking_type ==="
  echo "Using config file: $CONFIG_FILE"
  echo "Setting distance_threshold to $distance_threshold"
  echo "Setting post_score to $POST_SCORE"
  echo "Setting llm_output_data_file to $LLM_OUTPUT_FILE"
  
  # Update YAML config
  sed -i "s/tracking_type: .*/tracking_type: $tracking_type/" "$CONFIG_FILE"
  sed -i "s/distance_threshold: .*/distance_threshold: $distance_threshold/" "$CONFIG_FILE"
  sed -i "s/post_score: .*/post_score: $POST_SCORE/" "$CONFIG_FILE"
  sed -i "s|llm_output_data_file: .*|llm_output_data_file: \"$LLM_OUTPUT_FILE\"|" "$CONFIG_FILE"
  
  # Run the Python tracking script
  python3 "$PYTHON_SCRIPT" --cfg_file "$CONFIG_FILE"
  
  echo "=== Finished $tracking_type ==="
} 


# === MAIN LOOP ===
for index in "${!DETECTIONS_PATHS[@]}"; do
  DETECTION_PATH="${DETECTIONS_PATHS[$index]}"
  LOG_FILE="${RESULTS_OUTPUT_TEXT[$index]}"

  echo ""
  echo "##############################"
  echo "Processing detection path: $DETECTION_PATH"
  echo "Logging to file: $LOG_FILE"
  echo "##############################"

  {
    # === Update CAR config file with detection path ===
    sed -i "s|detections_path: .*|detections_path: \"$DETECTION_PATH\"|" "$CAR_CONFIG_FILE"
    
    # === Update PEDESTRIAN config file with detection path ===
    sed -i "s|detections_path: .*|detections_path: \"$DETECTION_PATH\"|" "$PEDESTRIAN_CONFIG_FILE"

    # === Run for Car (distance_threshold 150) ===
    run_with_tracking_type "Car" 150 "$index"

    # === Run for Pedestrian (distance_threshold 180) ===
    run_with_tracking_type "Pedestrian" 180 "$index"

    # === EVALUATION AFTER BOTH RUNS ===
    echo ""
    echo "=============================="
    echo "Running evaluation script after Car and Pedestrian..."
    echo "=============================="

    cd TrackEval/scripts || exit
    bash "$EVAL_SCRIPT"
    cd ../../ || exit

    echo "=============================="
    echo "Evaluation complete for index $index!"
    echo "=============================="

  } &> "$LOG_FILE"  # Save both stdout and stderr to the log file

  echo "Logs for index $index saved to $LOG_FILE"
done

# === EVALUATION ===
echo ""
echo "=============================="
echo "Running evaluation script..."
echo "=============================="
cd TrackEval/scripts || exit
bash "$EVAL_SCRIPT"

# Return to root dir
cd ../../ || exit

echo "=============================="
echo "All runs and evaluations completed!"
echo "=============================="
