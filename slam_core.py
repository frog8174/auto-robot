import numpy as np
import math
from slam_config import MAP_SIZE_PIXELS, MAP_RESOLUTION, MAP_CENTER_X, MAP_CENTER_Y

class ExtendedKalmanFilter:
    def __init__(self, dt, init_pose=None):
        self.dt = dt
        # State vector [x, y, theta]
        self.x = np.array([0.0, 0.0, 0.0]) if init_pose is None else init_pose
        
        # Covariance Matrix P (Uncertainty)
        # High uncertainty initially
        self.P = np.eye(3) * 0.1
        
        # Process Noise Covariance Q (Model uncertainty)
        # x/y uncertainty usually comes from slip, theta from gyro drift/slip
        self.Q = np.diag([0.01, 0.01, 0.005]) 
        
        # Measurement Noise Covariance R (Sensor uncertainty)
        # We only measure Theta rate (Gyro) effectively here in this simple setup
        # or we treat the Gyro integration as a "measurement" of theta change.
        # Let's formulate:
        # Prediction: x_k = f(x_{k-1}, u_k)  (u = v, w from motors)
        # Correction: We observe angular velocity from Gyro. 
        # But standard EKF usually corrects Position with GPS/Lidar. 
        # Here we only have Gyro.
        
        # Strategy: Fusing Motor Odom (w) and Gyro (w_gyro) for Theta.
        # We will keep it simple: The prediction uses Motor V, and a weighted blend of Motor W and Gyro W.
        # Actually, a formal EKF might be overkill if we don't have absolute position updates, 
        # but it helps track uncertainty P.
        pass

    def predict_update(self, v_cmd, w_cmd, gyro_z, dt):
        """
        Since we lack absolute position sensors (GPS/Lidar matching in this step),
        we perform a 'Loose Coupling' fusion for Heading.
        """
        # 1. Prediction Step (based on Control Command)
        theta = self.x[2]
        
        # Jacobians (Linearization)
        # x' = x + v*cos(theta)*dt
        # y' = y + v*sin(theta)*dt
        # th' = th + w*dt
        
        # F = df/dx
        F = np.eye(3)
        F[0, 2] = -v_cmd * math.sin(theta) * dt
        F[1, 2] =  v_cmd * math.cos(theta) * dt
        
        # Predict State
        self.x[0] += v_cmd * math.cos(theta) * dt
        self.x[1] += v_cmd * math.sin(theta) * dt
        self.x[2] += w_cmd * dt
        
        # Predict Covariance
        self.P = F @ self.P @ F.T + self.Q
        
        # 2. Update Step (Correction using Gyro)
        # Only if we have valid gyro data
        if gyro_z is not None:
            # We treat Gyro as measuring the *rate*. 
            # Innovation: z - Hx
            # But standard EKF requires z to be state. 
            # Let's simplify: We use a 1D Kalman Filter logic for Theta implicitly.
            # Or, we treat (Theta_prev + Gyro*dt) as a "Measurement" of Theta.
            
            z_theta = theta + gyro_z * dt # The 'measured' new theta based on Gyro
            
            # H = Measurement matrix (We are observing Theta directly-ish)
            H = np.array([[0, 0, 1]])
            
            # Measurement Noise (Gyro integration error)
            R = np.array([[0.01]]) 
            
            # Kalman Gain
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)
            
            # Update State
            y = z_theta - self.x[2] # Innovation
            
            # Normalize angle innovation
            y = math.atan2(math.sin(y), math.cos(y))
            
            self.x = self.x + (K @ np.array([y])) # Simple 1D vector mult hack
            
            # Update Covariance
            I = np.eye(3)
            self.P = (I - K @ H) @ self.P
            
        # Normalize Theta final
        self.x[2] = math.atan2(math.sin(self.x[2]), math.cos(self.x[2]))
        
        return self.x

class OccupancyGridSLAM:
    def __init__(self):
        # Pose: [x (meters), y (meters), theta (radians)]
        self.pose = np.array([0.0, 0.0, 0.0])
        self.ekf = ExtendedKalmanFilter(dt=0.1)
        
        # Map: Log-odds representation
        # 0 = Unknown (log-odds 0)
        # >0 = Occupied
        # <0 = Free
        self.map_logodds = np.zeros((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS))
        
        # Constants for Bresenham's algorithm / Map update
        self.l_occ = math.log(0.8 / 0.2)  # Log odds for occupied cell
        self.l_free = math.log(0.3 / 0.7) # Log odds for free cell
        self.max_occ = 10.0
        self.min_occ = -10.0
        
    def predict(self, v, w, dt, gyro_z_corrected=None):
        """
        Update pose using EKF.
        v: Linear velocity command (m/s)
        w: Angular velocity command (rad/s)
        dt: Time delta (s)
        gyro_z_corrected: IMU yaw rate (rad/s)
        """
        # Use EKF to update pose
        self.pose = self.ekf.predict_update(v, w, gyro_z_corrected, dt)
        return self.pose

    def update(self, distance_reading, sensor_angle_offset=0.0):
        """
        Update map based on single ultrasonic sensor reading.
        distance_reading: Distance in meters. If None or max range, we assume clear space up to a limit.
        sensor_angle_offset: Angle of the sensor relative to the robot's heading (radians).
        """
        MAX_RANGE = 2.0 # Max reliable range of ultrasonic in meters
        
        r = distance_reading
        if r is None or r > MAX_RANGE:
            r = MAX_RANGE # Just clear space, don't mark obstacle
            is_obstacle = False
        else:
            is_obstacle = True

        # Robot position in map grid coordinates
        x_start = int(self.pose[0] / MAP_RESOLUTION) + MAP_CENTER_X
        y_start = int(self.pose[1] / MAP_RESOLUTION) + MAP_CENTER_Y
        
        # Obstacle position in world coordinates
        # Total angle = Robot Heading + Sensor Offset
        theta = self.pose[2] + sensor_angle_offset
        ox = self.pose[0] + r * math.cos(theta)
        oy = self.pose[1] + r * math.sin(theta)
        
        # Obstacle position in map grid coordinates
        x_end = int(ox / MAP_RESOLUTION) + MAP_CENTER_X
        y_end = int(oy / MAP_RESOLUTION) + MAP_CENTER_Y
        
        # Ray tracing (Bresenham's Line Algorithm)
        cells = self._bresenham(x_start, y_start, x_end, y_end)
        
        # Update map
        for (cx, cy) in cells:
            if not (0 <= cx < MAP_SIZE_PIXELS and 0 <= cy < MAP_SIZE_PIXELS):
                continue
                
            # If this is the last cell and it was a hit, mark occupied
            if is_obstacle and (cx, cy) == (x_end, y_end):
                self.map_logodds[cy, cx] += self.l_occ
            else:
                # Mark as free space
                self.map_logodds[cy, cx] += self.l_free
                
        # Clamp values to prevent instability
        np.clip(self.map_logodds, self.min_occ, self.max_occ, out=self.map_logodds)

    def _bresenham(self, x0, y0, x1, y1):
        """Generates list of grid cells along the line (x0,y0) -> (x1,y1)."""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                cells.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                cells.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        cells.append((x, y))
        return cells

    def get_map_prob(self):
        """Returns map converted to probabilities (0.0 to 1.0)."""
        # Sigmoid function: p = 1 / (1 + exp(-log_odds))
        return 1.0 / (1.0 + np.exp(-self.map_logodds))

    def get_map_image(self):
        """Returns a uint8 image (0-255) for visualization. 0=Free, 127=Unknown, 255=Occupied."""
        # Convert probability 0..1 to pixel 255..0 (Black=Occupied, White=Free is standard map, 
        # but let's do: 0=Free, 255=Occupied for display)
        
        prob = self.get_map_prob()
        
        # Mapping:
        # Prob ~ 0.0 (Free) -> 255 (White)
        # Prob ~ 0.5 (Unknown) -> 127 (Gray)
        # Prob ~ 1.0 (Occupied) -> 0 (Black)
        img = (255 * (1 - prob)).astype(np.uint8)
        
        # Draw robot
        rx = int(self.pose[0] / MAP_RESOLUTION) + MAP_CENTER_X
        ry = int(self.pose[1] / MAP_RESOLUTION) + MAP_CENTER_Y
        
        # Simple marker for robot
        if 0 <= rx < MAP_SIZE_PIXELS and 0 <= ry < MAP_SIZE_PIXELS:
            import cv2
            cv2.circle(img, (rx, ry), 3, 100, -1) # Dark gray dot for robot
            
            # Heading line
            hx = int(rx + 10 * math.cos(self.pose[2]))
            hy = int(ry + 10 * math.sin(self.pose[2]))
            cv2.line(img, (rx, ry), (hx, hy), 100, 1)

        return img
