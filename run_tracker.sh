#!/bin/bash
# Define configuration files
CAR_CONFIG_FILE="config/global/cfg_refergpt.yaml"
PEDESTRIAN_CONFIG_FILE="config/global/cfg_refergpt_pedestrian.yaml"
PYTHON_SCRIPT="main.py"
EVAL_SCRIPT="evaluate_cliptrack.sh"

# Function to update tracking_type and run the script
run_with_tracking_type() {
  local tracking_type="$1"  # First argument is the tracking_type
  local distance_threshold="$2"
  
  # Select the appropriate configuration file
  if [ "$tracking_type" = "Car" ] 
  then
    CONFIG_FILE="$CAR_CONFIG_FILE"
  elif [ "$tracking_type" = "Pedestrian" ]  
  then
    CONFIG_FILE="$PEDESTRIAN_CONFIG_FILE"
  else
    echo "Unknown tracking type: $tracking_type"
    exit 1
  fi
  
  echo "Running with tracking_type: $tracking_type using config: $CONFIG_FILE"
  
  # Update the tracking_type and distance_threshold in the YAML file
  sed -i "s/tracking_type: .*/tracking_type: $tracking_type/" "$CONFIG_FILE"
  sed -i "s/distance_threshold: .*/distance_threshold: $distance_threshold/" "$CONFIG_FILE"
  
  # Run the Python script
  python3 "$PYTHON_SCRIPT" --cfg_file "$CONFIG_FILE"
  echo "Finished tracking with $tracking_type"
}

# # Run for "Car"
run_with_tracking_type "Car" 150

# # Run for "Pedestrian"
run_with_tracking_type "Pedestrian" 180

# Run evaluation script
echo "Running evaluation script..."
cd TrackEval/scripts || exit
bash "$EVAL_SCRIPT"

# Return to the original directory
echo "Returning to the original directory..."
cd ../../ || exit

echo "All runs and evaluations completed!"