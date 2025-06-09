"""
trajectory_utils.py
Utilities for trajectory processing and movement description generation.
"""
def generate_movement_description(car_trajectory, image, detection_2d):
    """Generate a movement description for a trajectory (stub for LLM or rule-based)."""
    # This function can be extended to use an LLM or rule-based system
    # For now, just return a string summary

    
    desc = (
        f"Object at (x={car_trajectory['x=']:.2f}, y={car_trajectory['y=']:.2f}, z={car_trajectory['z=']:.2f}) "
        f"moved {car_trajectory['euclidean_distance_difference=']:.2f}m, "
        f"heading {car_trajectory['average_direction_angle=']:.2f} rad, "
        f"mean heading variation {car_trajectory['mean_heading_variation=']:.2f}, "
        f"x_var={car_trajectory['x_variation=']:.2f}, y_var={car_trajectory['y_variation=']:.2f}, z_var={car_trajectory['z_variation=']:.2f}."
    )
    return desc
