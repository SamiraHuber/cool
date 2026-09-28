std_msgs/Header header

int32 instance_id
int32 track_id
int32 class_id
string class_name
float32 score
bool detected_in_previous_frame

int32 x_min
int32 y_min
int32 x_max
int32 y_max

sensor_msgs/Image cropped_image
sensor_msgs/Image original_cropped_image
sensor_msgs/Image mask_image