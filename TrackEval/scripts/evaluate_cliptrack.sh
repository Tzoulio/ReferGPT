# python3 run_mot_challenge.py \
# --METRICS HOTA \
# --SEQMAP_FILE dataset/data_path_v2/seqmap.txt \
# --SKIP_SPLIT_FOL True \
# --GT_FOLDER dataset/data/KITTI/tracking/training/image_02 \
# --TRACKERS_FOLDER evaluation/results/sha_key/data/refer-kitti-v2 \
# --GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
# --TRACKERS_TO_EVAL evaluation/results/sha_key/data/refer-kitti-v2 \
# --USE_PARALLEL True \
# --NUM_PARALLEL_CORES 2 \
# --SKIP_SPLIT_FOL True \
# --PLOT_CURVES False

python3 run_mot_challenge.py \
--ROOT_DIR ~/Documents/TrackGPT/ClipTrack \
--METRICS HOTA \
--SEQMAP_FILE dataset/data_path/seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER dataset/data/KITTI/tracking/training/image_02 \
--TRACKERS_FOLDER evaluation/results/sha_key/data/refer-kitti \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL evaluation/results/sha_key/data/refer-kitti \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--SKIP_SPLIT_FOL True \
--PLOT_CURVES False

