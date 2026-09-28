# Source ROS
```
source /opt/ros/humble/setup.sh
source /workspace/install/setup.sh || true
```

# Create Network (should be done in a docker compose file soon)
```
docker network create bordsupr
```

```
docker network connect bordsupr busy_mclaren
docker network connect bordsupr pgvector-db_3
docker network connect bordsupr frontend-web-1
```

```
docker network inspect bordsupr
```

Test:
```
python3 -c "import psycopg; psycopg.connect(host='pgvector-db', port=5432, dbname='bordsupr', user='postgres', password='postgres').close(); print('connected')"
```

# Build packages
```
colcon build --symlink-install
source install/setup.bash
```

## Or to build single package
```
colcon build --packages-select bordsupr_interfaces bordsupr --symlink-install
source install/setup.bash
```

## Check if package is built
```
ros2 pkg list | grep bordsupr
ros2 pkg executables bordsupr
```

# Start ROS2 Spot Driver
```
# Build config from env (SPOT_USERNAME / SPOT_PASSWORD must be set, see template.env)
: "${SPOT_HOSTNAME:=192.168.80.3}"
: "${SPOT_USERNAME:?set SPOT_USERNAME}"
: "${SPOT_PASSWORD:?set SPOT_PASSWORD}"

cat >/workspace/spot_config.yaml <<'EOF'
/**:
  ros__parameters:
    hostname: "__HOST__"
    username: "__USER__"
    password: "__PASS__"
    use_lease: false
    estop_timeout: 9.0
    publish_point_cloud: true       # 👈 make sure this is set
    publish_depth_images: true      # optional, if you want RGBD camera clouds
    lidar_source: "velodyne-point-cloud"        # or "depth", depending on what you have
    # Add further driver params here if needed
EOF

# Inject values safely
sed -i "s|__HOST__|${SPOT_HOSTNAME}|g" /workspace/spot_config.yaml
sed -i "s|__USER__|${SPOT_USERNAME}|g" /workspace/spot_config.yaml
sed -i "s|__PASS__|${SPOT_PASSWORD}|g" /workspace/spot_config.yaml

echo "==== spot_config.yaml ===="
cat /workspace/spot_config.yaml
echo "=========================="


ros2 launch spot_driver spot_driver.launch.py \
  config_file:=/workspace/spot_config.yaml \
  launch_image_publishers:=true \
  publish_compressed_images:=true \
  launch_rviz:=false &
```

# Launch Custom package
```
ros2 launch bordsupr bordsupr.launch.py
```