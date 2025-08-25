import numpy as np
import math
from typing import Dict, Tuple, Union, List, Optional

from sim_launcher import simulation_app  # garante que o SimulationApp foi inicializado antes

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.cloner import GridCloner
from pxr import Sdf, UsdLux, UsdGeom
import isaacsim.core.utils.stage as stage_utils
import omni.usd

# Import Isaac Sim transform utilities
try:
    from nvidia.srl.math.transform import Transform
except ImportError:
    # Fallback for different Isaac Sim versions
    try:
        from isaacsim.core.utils.math import Transform
    except ImportError:
        print("[INFO] Using manual quaternion conversion (Transform utilities not available)")
        # This is normal and doesn't affect functionality
        Transform = None

# Import rotation utilities
try:
    from isaacsim.core.utils.rotations import quat_to_euler_angles
except ImportError:
    print("[INFO] Rotation utilities not available - using fallback methods")
    quat_to_euler_angles = None

class RobotObserver:
    """
    Observer class for Go2 robot providing comprehensive monitoring capabilities.
    Handles both robot-level and joint-level observations for RL environments.
    
    Uses Isaac Sim's native Transform utilities for robust quaternion/Euler conversions
    and transformation matrix operations when available.
    """
    
    def __init__(self, articulation, joint_names=None, robot_index=0):
        """
        Initialize the robot observer.
        
        Args:
            articulation: Go2 robot Articulation object (can be multi-robot view)
            joint_names: Optional list of joint names for easier indexing
            robot_index: Index of the robot in multi-robot articulation (default: 0)
        """
        self.robot = articulation
        self.robot_index = robot_index  # Which robot in the articulation view
        
        # Store robot properties
        self.num_dof = self.robot.num_dof
        
        # Initialize joint limits - com verificação de segurança
        self.joint_limits = None
        self._initialize_joint_limits()
        
        # Joint names mapping (Go2 specific)
        if joint_names is None:
            self.joint_names = [
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", 
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
            ]
        else:
            self.joint_names = joint_names
            
        # Initialize observation history for velocity calculations
        self.prev_position = None
        self.prev_time = None
    
    def _initialize_joint_limits(self):
        """
        Initialize joint limits with safety checks and fallbacks.
        """
        try:
            self.joint_limits = self.robot.get_dof_limits()
            
            if self.joint_limits is None:
                print("[WARNING] RobotObserver: joint_limits is None, using fallback values")
                self._set_fallback_joint_limits()
            elif len(self.joint_limits.shape) == 0:
                print("[WARNING] RobotObserver: joint_limits has invalid shape, using fallback values")
                self._set_fallback_joint_limits()
            else:
                print(f"[INFO] RobotObserver: joint_limits initialized with shape {self.joint_limits.shape}")
                
        except Exception as e:
            print(f"[ERROR] RobotObserver: Failed to get joint limits: {e}")
            self._set_fallback_joint_limits()
    
    def _set_fallback_joint_limits(self):
        """
        Set fallback joint limits for Go2 robot when real limits are not available.
        """
        print("[INFO] RobotObserver: Using fallback joint limits for Go2")
        
        # Go2 typical joint limits (conservative values)
        hip_limit = np.radians(45)      # ±45 degrees for hip
        thigh_lower = np.radians(-45)   # -45 degrees for thigh
        thigh_upper = np.radians(90)    # +90 degrees for thigh  
        calf_lower = np.radians(-160)   # -160 degrees for calf
        calf_upper = np.radians(-30)    # -30 degrees for calf
        
        # Create limits array for 12 joints
        lower_limits = np.array([
            -hip_limit, thigh_lower, calf_lower,  # FL
            -hip_limit, thigh_lower, calf_lower,  # FR
            -hip_limit, thigh_lower, calf_lower,  # RL
            -hip_limit, thigh_lower, calf_lower   # RR
        ])
        
        upper_limits = np.array([
            hip_limit, thigh_upper, calf_upper,   # FL
            hip_limit, thigh_upper, calf_upper,   # FR
            hip_limit, thigh_upper, calf_upper,   # RL
            hip_limit, thigh_upper, calf_upper    # RR
        ])
        
        # Create joint_limits in the expected format [num_envs, num_joints, 2]
        # For now, assume single environment, will be expanded if needed
        self.joint_limits = np.array([[lower_limits, upper_limits]]).transpose(0, 2, 1)
        print(f"[INFO] RobotObserver: Fallback joint_limits shape: {self.joint_limits.shape}")

    def get_robot_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get robot base position and orientation.
        
        Returns:
            Tuple of (position [x,y,z], orientation_quat [x,y,z,w])
        """
        positions, orientations = self.robot.get_world_poses()
        return np.array(positions[self.robot_index]), np.array(orientations[self.robot_index])
        
    def get_robot_velocities(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get robot base linear and angular velocities.
        
        Returns:
            Tuple of (linear_velocity [x,y,z], angular_velocity [x,y,z])
        """
        lin_vel = self.robot.get_linear_velocities()
        ang_vel = self.robot.get_angular_velocities()
        return np.array(lin_vel[self.robot_index]), np.array(ang_vel[self.robot_index])

    def _manual_quat_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """
        Manual quaternion to Euler conversion as fallback.
        
        Args:
            quat: Quaternion [w, x, y, z]
            
        Returns:
            Euler angles [roll, pitch, yaw] in radians
        """
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = math.sqrt(1 + 2 * (w * y - z * x))
        cosp = math.sqrt(1 -2 * (w * y - z * x))
        pitch = 2 * math.atan2(sinp, cosp) - math.pi / 2
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])

    def quaternion_to_euler(self, quat: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).
        
        Args:
            quat: Quaternion [w, x, y, z]
            
        Returns:
            Euler angles [roll, pitch, yaw] in radians
        """
        if quat_to_euler_angles is not None:
            return quat_to_euler_angles(quat, degrees=False, extrinsic=False)
        else:
            # Fallback manual conversion if import failed
            return self._manual_quat_to_euler(quat)
        
    def get_robot_orientation_euler(self) -> np.ndarray:
        """
        Get robot orientation as Euler angles.
        
        Returns:
            Euler angles [roll, pitch, yaw] in radians
        """
        _, quat = self.get_robot_pose()
        return self.quaternion_to_euler(quat)
        
    def get_joint_states(self) -> Dict[str, np.ndarray]:
        """
        Get all joint positions and velocities.
        
        Returns:
            Dictionary with 'positions' and 'velocities' arrays
        """
        joint_state = self.robot.get_joints_state()
        return {
            'positions': joint_state.positions[self.robot_index],
            'velocities': joint_state.velocities[self.robot_index]
        }
        
    def get_comprehensive_observation(self) -> Dict[str, Union[np.ndarray, float]]:
        """
        Get comprehensive observation state for RL.
        
        Returns:
            Dictionary containing all relevant observations
        """
        # Robot pose and motion
        position, quat = self.get_robot_pose()
        euler = self.quaternion_to_euler(quat)
        lin_vel, ang_vel = self.get_robot_velocities()
        
        # Joint states
        joint_states = self.get_joint_states()
        joint_pos_normalized = self.normalize_joint_positions(joint_states['positions'])
        
        # Computed metrics
        forward_velocity = self.get_forward_velocity()
        orientation_penalty = self.compute_orientation_penalty()
        
        observation = {
            # Robot base state
            'robot_position': position,
            'robot_orientation_quat': quat,
            'robot_orientation_euler': euler,
            'robot_linear_velocity': lin_vel,
            'robot_angular_velocity': ang_vel,
            'forward_velocity': forward_velocity,
            
            # Joint states
            'joint_positions': joint_states['positions'],
            'joint_velocities': joint_states['velocities'],
            'joint_positions_normalized': joint_pos_normalized,
            
            # Computed metrics for rewards
            'orientation_penalty': orientation_penalty,
            'height': position[2],  # Z-coordinate as height
        }
        
        return observation

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get joint limits for the specific robot in physical units (radians).
        
        Returns:
            Tuple of (lower_limits, upper_limits) for this robot in radians
        """
        # Ensure joint_limits are available
        if self.joint_limits is None:
            print("[WARNING] RobotObserver: joint_limits is None, reinitializing...")
            self._initialize_joint_limits()
            
        if self.joint_limits is None:
            print("[ERROR] RobotObserver: Failed to initialize joint_limits, using zero limits")
            return np.zeros(12), np.zeros(12)
        
        try:
            if len(self.joint_limits.shape) == 3:  # Multi-robot view
                lower = self.joint_limits[self.robot_index, :, 0]
                upper = self.joint_limits[self.robot_index, :, 1]
            else:  # Single robot or fallback format
                if self.joint_limits.shape[1] == 2:  # [num_joints, 2] format
                    lower = self.joint_limits[:, 0]
                    upper = self.joint_limits[:, 1]
                else:  # [2, num_joints] format (our fallback)
                    lower = self.joint_limits[0, :]
                    upper = self.joint_limits[1, :]
                    
            return lower, upper
            
        except Exception as e:
            print(f"[ERROR] RobotObserver: Error getting joint limits: {e}")
            print(f"[ERROR] RobotObserver: joint_limits shape: {self.joint_limits.shape if self.joint_limits is not None else 'None'}")
            # Return safe default limits
            return np.full(12, -1.0), np.full(12, 1.0)

    def normalize_joint_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Normalize joint positions to [-1, 1] range based on joint limits.
        
        Args:
            positions: Joint positions in radians
            
        Returns:
            Normalized joint positions
        """
        lower, upper = self.get_joint_limits()
        normalized = 2.0 * (positions - lower) / (upper - lower) - 1.0
        return np.clip(normalized, -1.0, 1.0)

    def get_forward_velocity(self) -> float:
        """
        Get forward velocity in robot's local coordinate frame.
        
        Returns:
            Forward velocity (m/s)
        """
        position, quat = self.get_robot_pose()
        lin_vel, _ = self.get_robot_velocities()
        
        # Convert quaternion to rotation matrix
        transform_matrix = self.get_robot_transform_matrix()
        
        # Transform velocity to robot's local frame
        if transform_matrix is not None:
            # Extract rotation part (3x3)
            rotation_matrix = transform_matrix[:3, :3]
            local_velocity = rotation_matrix.T @ lin_vel
            forward_velocity = local_velocity[0]  # X-axis is forward
        else:
            # Fallback: use yaw angle only
            euler = self.quaternion_to_euler(quat)
            yaw = euler[2]
            forward_velocity = (lin_vel[0] * math.cos(yaw) + 
                              lin_vel[1] * math.sin(yaw))
        
        return forward_velocity

    def get_robot_transform_matrix(self) -> np.ndarray:
        """
        Get 4x4 transformation matrix for the robot.
        
        Returns:
            4x4 transformation matrix or None if not available
        """
        if Transform is None:
            return None
            
        try:
            position, quat = self.get_robot_pose()
            
            # Create transform object
            transform = Transform()
            transform.translation = position
            transform.orientation = quat
            
            # Get transformation matrix
            transform_matrix = transform.transformation_matrix
            return transform_matrix
            
        except Exception as e:
            print(f"Warning: Could not compute transform matrix: {e}")
            return None

    def compute_orientation_penalty(self) -> float:
        """
        Compute penalty for robot orientation (tilt from upright).
        
        Returns:
            Orientation penalty value
        """
        euler = self.get_robot_orientation_euler()
        roll, pitch, yaw = euler
        
        # Penalty based on tilt magnitude
        tilt_magnitude = math.sqrt(roll**2 + pitch**2)
        
        # Exponential penalty for excessive tilt
        if tilt_magnitude > 0.5:  # 28.6 degrees
            penalty = (tilt_magnitude - 0.5) * 2.0
        else:
            penalty = tilt_magnitude**2
            
        return penalty


class Go2RewardCalculator:
    """
    Advanced reward calculator for Go2 robot RL training.
    Provides various reward components and penalty functions.
    """
    
    def __init__(self, observer: RobotObserver):
        """
        Initialize reward calculator.
        
        Args:
            observer: RobotObserver instance
        """
        self.observer = observer
        self.prev_obs = None
        
        # Reward weights (can be tuned)
        self.weights = {
            'velocity': 1.0,
            'orientation': 0.5,
            'action_penalty': 0.01,
            'energy_penalty': 0.001,
            'stability': 0.3,
            'survival': 0.1,
            'smoothness': 0.2,
            'height': 0.2
        }
        
    def compute_velocity_reward(self, target_velocity: float = 1.0, 
                               velocity_tolerance: float = 0.1) -> Dict[str, float]:
        """
        Compute reward based on forward velocity tracking.
        
        Args:
            target_velocity: Desired forward velocity
            velocity_tolerance: Tolerance for perfect velocity reward
            
        Returns:
            Dictionary with velocity reward components
        """
        forward_vel = self.observer.get_forward_velocity()
        velocity_error = abs(forward_vel - target_velocity)
        
        # Exponential reward for being close to target
        if velocity_error <= velocity_tolerance:
            velocity_reward = 1.0
        else:
            velocity_reward = math.exp(-2.0 * velocity_error)
            
        # Bonus for maintaining target velocity
        velocity_bonus = 0.0
        if velocity_error <= velocity_tolerance * 0.5:
            velocity_bonus = 0.2
            
        return {
            'velocity_reward': velocity_reward,
            'velocity_bonus': velocity_bonus,
            'velocity_error': velocity_error,
            'forward_velocity': forward_vel
        }
        
    def compute_orientation_reward(self, max_tilt: float = 0.3) -> Dict[str, float]:
        """
        Compute reward for maintaining upright orientation.
        
        Args:
            max_tilt: Maximum allowed tilt in radians
            
        Returns:
            Dictionary with orientation reward components
        """
        euler = self.observer.get_robot_orientation_euler()
        roll, pitch, yaw = euler
        
        # Penalize roll and pitch (tilt)
        tilt_magnitude = math.sqrt(roll**2 + pitch**2)
        
        if tilt_magnitude <= max_tilt:
            orientation_reward = 1.0 - (tilt_magnitude / max_tilt)**2
        else:
            orientation_reward = 0.0
            
        # Additional penalty for extreme tilts
        extreme_tilt_penalty = 0.0
        if tilt_magnitude > max_tilt * 1.5:
            extreme_tilt_penalty = (tilt_magnitude - max_tilt * 1.5) * 2.0
            
        return {
            'orientation_reward': orientation_reward,
            'extreme_tilt_penalty': extreme_tilt_penalty,
            'tilt_magnitude': tilt_magnitude,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw
        }
        
    def compute_height_reward(self, target_height: float = 0.5, 
                             height_tolerance: float = 0.1) -> Dict[str, float]:
        """
        Compute reward for maintaining target height.
        
        Args:
            target_height: Desired robot height
            height_tolerance: Tolerance for height variation
            
        Returns:
            Dictionary with height reward components
        """
        obs = self.observer.get_comprehensive_observation()
        current_height = obs['height']
        height_error = abs(current_height - target_height)
        
        if height_error <= height_tolerance:
            height_reward = 1.0 - (height_error / height_tolerance)**2
        else:
            height_reward = 0.0
            
        # Penalty for being too low (potential fall)
        fall_penalty = 0.0
        if current_height < 0.22:  # Updated to match SimHelper changes
            fall_penalty = (0.22 - current_height) * 10.0  # Increased penalty
            
        return {
            'height_reward': height_reward,
            'fall_penalty': fall_penalty,
            'height_error': height_error,
            'current_height': current_height
        }

    def compute_comprehensive_reward(self, target_velocity: float = 1.0,
                                   target_height: float = 0.5,
                                   current_actions: np.ndarray = None,
                                   custom_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Compute comprehensive reward combining all components.
        
        Args:
            target_velocity: Target forward velocity
            target_height: Target robot height
            current_actions: Current joint commands (optional)
            custom_weights: Custom weights for reward components
            
        Returns:
            Dictionary with all reward components and total reward
        """
        # Use custom weights if provided
        weights = self.weights.copy()
        if custom_weights:
            weights.update(custom_weights)
            
        # Compute individual reward components
        velocity_info = self.compute_velocity_reward(target_velocity)
        orientation_info = self.compute_orientation_reward()
        height_info = self.compute_height_reward(target_height)
        
        # Stability reward (low angular velocities)
        _, angular_vel = self.observer.get_robot_velocities()
        angular_speed = np.linalg.norm(angular_vel)
        stability_reward = math.exp(-angular_speed * 2.0)
        
        # Combine all rewards
        total_reward = (
            velocity_info['velocity_reward'] * weights['velocity'] +
            velocity_info['velocity_bonus'] * weights['velocity'] +
            orientation_info['orientation_reward'] * weights['orientation'] +
            height_info['height_reward'] * weights['height'] +
            stability_reward * weights['stability'] -
            orientation_info['extreme_tilt_penalty'] * weights['orientation'] -
            height_info['fall_penalty'] * weights['height']
        )
        
        # Survival bonus (small reward for not terminating)
        survival_bonus = weights['survival']
        total_reward += survival_bonus
        
        # Compile comprehensive result
        result = {
            'total_reward': total_reward,
            'velocity_reward': velocity_info['velocity_reward'],
            'velocity_bonus': velocity_info['velocity_bonus'],
            'orientation_reward': orientation_info['orientation_reward'],
            'height_reward': height_info['height_reward'],
            'stability_reward': stability_reward,
            'survival_bonus': survival_bonus,
            'extreme_tilt_penalty': orientation_info['extreme_tilt_penalty'],
            'fall_penalty': height_info['fall_penalty'],
            'forward_velocity': velocity_info['forward_velocity'],
            'current_height': height_info['current_height'],
            'tilt_magnitude': orientation_info['tilt_magnitude']
        }
        
        return result

    def is_terminal_state(self, reward_info: Dict[str, float] = None) -> Tuple[bool, str]:
        """
        Check if the episode should terminate.
        
        Args:
            reward_info: Optional reward information from compute_comprehensive_reward
            
        Returns:
            Tuple of (should_terminate, termination_reason)
        """
        obs = self.observer.get_comprehensive_observation()
        
        # Check for fall
        if obs['height'] < 0.22:  # Updated to match SimHelper changes
            return True, "robot_fell"
            
        # Check for extreme tilt
        if obs['orientation_penalty'] > 2.0:
            return True, "extreme_tilt"
            
        # Check for extreme velocities (runaway behavior)
        linear_speed = np.linalg.norm(obs['robot_linear_velocity'])
        if linear_speed > 10.0:
            return True, "excessive_speed"
            
        return False, "continuing"


class MultiSimHelper:
    """
    Helper class for managing multiple Go2 robots in Isaac Sim for RL training.
    
    This class manages multi-robot simulation using GridCloner, including world creation,
    robot loading, action application, observation collection, and reward calculation
    for reinforcement learning with multiple parallel environments.
    """
    
    def __init__(self, num_envs: int = 4, spacing: float = 2.0, safety_margin: float = 0.1):
        """
        Initialize the multi-robot simulation helper.
        
        Args:
            num_envs: Number of robot environments to create
            spacing: Distance between robot environments in meters
            safety_margin: Safety margin for joint limits (0.1 = 10%)
        """
        self.num_envs = num_envs
        self.spacing = spacing
        self.safety_margin = safety_margin
        
        # Initialize world
        self.world = World(stage_units_in_meters=1.0)
        self.world.scene.add_default_ground_plane()
        
        # Add lighting
        stage = omni.usd.get_context().get_stage()
        UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight")).CreateIntensityAttr(300)
        
        # Setup multi-robot environment
        self._setup_multi_robot_environment()
        
        # Initialize simulation - IMPORTANTE: fazer antes de criar observers
        self.world.reset()
        
        # Calculate and store initial positions for each robot in the grid
        self._calculate_initial_positions()
        
        # Initialize observers and reward calculators APÓS o reset
        self.robot_observers = []
        self.reward_calculators = []
        
        # Get joint limits after reset - para garantir que estão disponíveis
        self.joint_limits = self.go2_robots.get_dof_limits()
        
        print(f"[INFO] MultiSimHelper: Joint limits shape after reset: {self.joint_limits.shape if self.joint_limits is not None else 'None'}")
        
        for i in range(self.num_envs):
            observer = RobotObserver(self.go2_robots, robot_index=i)
            # Force joint limits update in observer
            if self.joint_limits is not None:
                observer.joint_limits = self.joint_limits
            reward_calc = Go2RewardCalculator(observer)
            
            self.robot_observers.append(observer)
            self.reward_calculators.append(reward_calc)
        
        # Store baseline joint positions for each robot
        self.baseline_poses = []
        initial_joint_positions = self.go2_robots.get_joint_positions()
        if initial_joint_positions is not None:
            for i in range(self.num_envs):
                self.baseline_poses.append(initial_joint_positions[i].copy())
        else:
            print("[WARNING] MultiSimHelper: Could not get initial joint positions")
            # Create fallback baseline poses
            fallback_pose = np.zeros(12)  # 12 joints for Go2
            for i in range(self.num_envs):
                self.baseline_poses.append(fallback_pose.copy())
        
        # Set robots to their initial grid positions
        self._position_robots_in_grid()
            
        print(f"[INFO] MultiSimHelper initialized with {self.num_envs} environments")
        print(f"[INFO] Each robot has {self.go2_robots.num_dof} DOF")
        print(f"[INFO] Robot grid positions: {[f'({pos[0]:.1f}, {pos[1]:.1f})' for pos in self.initial_positions]}")

    def _calculate_initial_positions(self):
        """
        Calculate the initial grid positions for each robot based on spacing.
        """
        self.initial_positions = []
        
        # Calculate grid dimensions (try to make square grid)
        grid_size = int(np.ceil(np.sqrt(self.num_envs)))
        
        print(f"[INFO] Creating {grid_size}x{grid_size} grid for {self.num_envs} robots with spacing {self.spacing}m")
        
        for i in range(self.num_envs):
            # Calculate grid position (row, col)
            row = i // grid_size
            col = i % grid_size
            
            # Convert to world coordinates with spacing
            x = col * self.spacing - (grid_size - 1) * self.spacing / 2  # Center the grid
            y = row * self.spacing - (grid_size - 1) * self.spacing / 2  # Center the grid
            z = 0.5  # Height above ground
            
            self.initial_positions.append(np.array([x, y, z]))
        
        print(f"[INFO] Calculated initial positions for {self.num_envs} robots:")
        for i, pos in enumerate(self.initial_positions):
            print(f"  Robot {i}: ({pos[0]:6.2f}, {pos[1]:6.2f}, {pos[2]:4.2f})")

    def _position_robots_in_grid(self):
        """
        Position all robots at their initial grid positions.
        """
        if not hasattr(self, 'initial_positions'):
            print("[WARNING] Initial positions not calculated, using default")
            return
            
        # Get current poses
        current_world_poses = self.go2_robots.get_world_poses()
        positions, orientations = current_world_poses
        
        # Set each robot to its initial position
        for i in range(self.num_envs):
            if i < len(self.initial_positions):
                positions[i] = self.initial_positions[i].copy()
            else:
                # Fallback for extra robots
                positions[i] = np.array([0.0, 0.0, 0.5])
        
        # Apply new positions
        self.go2_robots.set_world_poses(positions, orientations)
        print(f"[INFO] Positioned {self.num_envs} robots in grid formation")

    def _setup_multi_robot_environment(self):
        """Setup the multi-robot environment using GridCloner."""
        
        # Load Go2 robot USD file
        assets_root_path = get_assets_root_path()
        go2_usd_path = assets_root_path + "/Isaac/Robots/Unitree/Go2/go2.usd"
        
        # Define base environment path
        env_zero_path = "/World/envs/env_0"
        
        print(f"[INFO] Loading Go2 robot from: {go2_usd_path}")
        
        # Add reference to the base environment
        stage_utils.add_reference_to_stage(usd_path=go2_usd_path, prim_path=f"{env_zero_path}/go2")
        
        # Create the base environment Xform
        UsdGeom.Xform.Define(stage_utils.get_current_stage(), env_zero_path)
        
        # Clone the environment using GridCloner
        print(f"[INFO] Cloning {self.num_envs} environments with spacing {self.spacing}m...")
        cloner = GridCloner(spacing=self.spacing)
        cloner.define_base_env(env_zero_path)
        
        # Generate environment paths and clone
        env_paths = cloner.generate_paths("/World/envs/env", self.num_envs)
        cloner.clone(source_prim_path=env_zero_path, prim_paths=env_paths)
        
        # Setup collision filtering
        # Sem essa função, a colisão entre os robôs existirá.
        physics_scene_path = self.world.get_physics_context().prim_path
        cloner.filter_collisions(
            physics_scene_path, 
            "/World/collisions", 
            env_paths, 
            global_paths=["/World/defaultGroundPlane"]
        )
        
        print(f"[INFO] Created environments at paths: {env_paths}")
        
        # Create articulation view for all robots
        self.go2_robots = self.world.scene.add(
            Articulation(
                prim_paths_expr="/World/envs/env.*/go2",
                name="go2_robot_view",
            )
        )
        
        print(f"[INFO] Created Articulation view with {self.go2_robots.count} robots")

    def apply_actions(self, actions: np.ndarray):
        """
        Apply actions to all robots with proper deserialization for absolute control.
        
        Args:
            actions: Actions array with shape (num_envs, num_joints) 
                    Values should be in [-1, 1] range and will be mapped to joint limits
        """
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"Actions shape {actions.shape} doesn't match num_envs {self.num_envs}")
        
        # Verify actions have the correct number of joints
        expected_joints = self.go2_robots.num_dof
        if actions.shape[1] != expected_joints:
            raise ValueError(f"Actions have {actions.shape[1]} joints, expected {expected_joints}")
            
        # Clip input actions to [-1, 1] range for safety
        actions_clipped = np.clip(actions, -1.0, 1.0)
        
        # Convert normalized actions [-1,1] to physical joint positions [lower,upper]
        joint_positions = np.zeros_like(actions_clipped)
        
        for env_id in range(self.num_envs):
            observer = self.robot_observers[env_id]
            lower, upper = observer.get_joint_limits()  # Get physical limits in radians
            
            # Desserialize [-1,1] -> [lower,upper]
            # Formula: pos = lower + (action + 1) / 2 * (upper - lower)
            joint_positions[env_id] = lower + (actions_clipped[env_id] + 1.0) / 2.0 * (upper - lower)
            
            # Safety clipping to ensure positions are within limits
            joint_positions[env_id] = np.clip(joint_positions[env_id], lower, upper)
        
        # Apply the converted joint positions
        try:
            self.go2_robots.set_joint_positions(joint_positions)
        except Exception as e:
            print(f"[ERROR] MultiSimHelper: Failed to apply actions: {e}")
            print(f"[ERROR] Actions shape: {actions.shape}")
            print(f"[ERROR] Joint positions shape: {joint_positions.shape}")
            print(f"[ERROR] Expected shape: ({self.num_envs}, {expected_joints})")
            raise

    def get_observations(self) -> List[Dict[str, Union[np.ndarray, float]]]:
        """
        Get observations from all robot environments.
        
        Returns:
            List of observation dictionaries, one per environment
        """
        observations = []
        for i in range(self.num_envs):
            try:
                obs = self.robot_observers[i].get_comprehensive_observation()
                observations.append(obs)
            except Exception as e:
                print(f"[ERROR] MultiSimHelper: Failed to get observation for env {i}: {e}")
                # Return a fallback observation
                fallback_obs = {
                    'robot_position': np.zeros(3),
                    'robot_orientation_quat': np.array([1, 0, 0, 0]),
                    'robot_orientation_euler': np.zeros(3),
                    'robot_linear_velocity': np.zeros(3),
                    'robot_angular_velocity': np.zeros(3),
                    'forward_velocity': 0.0,
                    'joint_positions': np.zeros(12),
                    'joint_velocities': np.zeros(12),
                    'joint_positions_normalized': np.zeros(12),
                    'orientation_penalty': 0.0,
                    'height': 0.5,
                }
                observations.append(fallback_obs)
        return observations

    def calculate_rewards(self, target_velocity: float = 1.0, 
                         target_height: float = 0.5) -> List[Dict[str, float]]:
        """
        Calculate rewards for all robot environments.
        
        Args:
            target_velocity: Target forward velocity for all robots
            target_height: Target height for all robots
            
        Returns:
            List of reward dictionaries, one per environment
        """
        rewards = []
        for i in range(self.num_envs):
            try:
                reward_info = self.reward_calculators[i].compute_comprehensive_reward(
                    target_velocity=target_velocity,
                    target_height=target_height
                )
                rewards.append(reward_info)
            except Exception as e:
                print(f"[ERROR] MultiSimHelper: Failed to calculate reward for env {i}: {e}")
                # Return a fallback reward
                fallback_reward = {
                    'total_reward': 0.0,
                    'velocity_reward': 0.0,
                    'velocity_bonus': 0.0,
                    'orientation_reward': 0.0,
                    'height_reward': 0.0,
                    'stability_reward': 0.0,
                    'survival_bonus': 0.0,
                    'extreme_tilt_penalty': 0.0,
                    'fall_penalty': 0.0,
                    'forward_velocity': 0.0,
                    'current_height': 0.5,
                    'tilt_magnitude': 0.0
                }
                rewards.append(fallback_reward)
        return rewards

    def check_terminations(self) -> List[Tuple[bool, str]]:
        """
        Check termination conditions for all environments.
        
        Returns:
            List of (should_terminate, reason) tuples
        """
        terminations = []
        for i in range(self.num_envs):
            try:
                terminal, reason = self.reward_calculators[i].is_terminal_state()
                terminations.append((terminal, reason))
            except Exception as e:
                print(f"[ERROR] MultiSimHelper: Failed to check termination for env {i}: {e}")
                # Return safe default (not terminated)
                terminations.append((False, "error_in_check"))
        return terminations

    def reset_environment(self, env_id: int):
        """
        Reset a specific environment to its initial grid position.
        
        Args:
            env_id: Index of environment to reset
        """
        if env_id >= self.num_envs:
            raise ValueError(f"Environment ID {env_id} >= num_envs {self.num_envs}")
            
        # Reset joint positions to baseline pose
        current_positions = self.go2_robots.get_joint_positions()
        if current_positions is not None and env_id < len(self.baseline_poses):
            current_positions[env_id] = self.baseline_poses[env_id].copy()
            self.go2_robots.set_joint_positions(current_positions)
        
        # Reset robot to its ORIGINAL grid position (not origin!)
        current_world_poses = self.go2_robots.get_world_poses()
        positions, orientations = current_world_poses
        
        if hasattr(self, 'initial_positions') and env_id < len(self.initial_positions):
            # Reset to original grid position with small random variation
            original_pos = self.initial_positions[env_id].copy()
            
            # Add small random offset for variation (only X and Y, keep Z stable)
            random_offset = np.random.normal(0, 0.05, 3)  # Smaller offset
            random_offset[2] = 0  # No random Z offset
            
            new_position = original_pos + random_offset
            new_position[2] = max(new_position[2], 0.3)  # Ensure minimum height
            
            positions[env_id] = new_position
            
            print(f"[INFO] Reset robot {env_id} to grid position ({new_position[0]:.2f}, {new_position[1]:.2f}, {new_position[2]:.2f})")
        else:
            # Fallback to origin if initial positions not available
            print(f"[WARNING] No initial position for robot {env_id}, using origin")
            positions[env_id] = np.array([0.0, 0.0, 0.5])
        
        # Reset orientation to default (upright)
        orientations[env_id] = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        
        self.go2_robots.set_world_poses(positions, orientations)

    def reset_all_environments(self):
        """Reset all environments to their initial grid positions."""
        print("[INFO] Resetting all environments to initial grid positions...")
        
        # Reset world simulation
        self.world.reset()
        
        # Reset all joint positions to baseline
        if len(self.baseline_poses) > 0:
            baseline_positions = np.array(self.baseline_poses)
            self.go2_robots.set_joint_positions(baseline_positions)
        
        # Reset all robots to their original grid positions
        self._position_robots_in_grid()
        
        print(f"[INFO] All {self.num_envs} robots reset to grid formation")

    def step_simulation(self, render: bool = True):
        """
        Step the simulation forward.
        
        Args:
            render: Whether to render the simulation
        """
        self.world.step(render=render)

    def log_environment_states(self, step: int, detailed: bool = False):
        """
        Log the state of all environments.
        
        Args:
            step: Current simulation step
            detailed: Whether to include detailed information
        """
        print(f"\n=== Multi-Environment States - Step {step} ===")
        
        observations = self.get_observations()
        rewards = self.calculate_rewards()
        
        for i in range(self.num_envs):
            obs = observations[i]
            reward_info = rewards[i]
            
            print(f"Env {i:2d}: Pos=[{obs['robot_position'][0]:5.2f}, {obs['robot_position'][1]:5.2f}, {obs['robot_position'][2]:5.2f}] "
                  f"Vel={obs['forward_velocity']:5.2f} m/s Reward={reward_info['total_reward']:6.3f}")
            
            if detailed and i == 0:  # Show details for first environment only
                print(f"  Detailed breakdown:")
                print(f"    Velocity reward: {reward_info['velocity_reward']:.3f}")
                print(f"    Orientation reward: {reward_info['orientation_reward']:.3f}")
                print(f"    Height reward: {reward_info['height_reward']:.3f}")
                print(f"    Stability reward: {reward_info['stability_reward']:.3f}")

    def close(self):
        """Close the simulation."""
        simulation_app.close()
