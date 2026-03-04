# Franka Environment Technical Reference

This document provides detailed technical information about the Franka Panda single-arm simulation environment (`pefm_envs/sim_franka/`).

## Architecture Overview

### PyBullet Setup

The environment uses PyBullet for physics simulation:

```python
# In franka_env.py _init_sim()
mode = pybullet.GUI if self.vis else pybullet.DIRECT
self.sim = bclient.BulletClient(connection_mode=mode)
```

- **DIRECT mode**: Headless simulation for batch demo generation
- **GUI mode**: Real-time visualization via `--vis` flag

GUI configuration:
```python
self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)      # Hide UI panels
self.sim.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS, 1)  # Enable shadows
self.sim.resetDebugVisualizerCamera(
    cameraDistance=1.2, cameraYaw=45, cameraPitch=-30,
    cameraTargetPosition=[0.4, 0, 0.1],
)
```

### Robot Configuration

Robot loaded from `pybullet_data`:
```python
# From pefm_envs/sim_mobile/utils/info.py
SIM_ROBOT_INFO["franka_panda"] = {
    "file_name": "franka_panda/panda.urdf",  # from pybullet_data
    "ee_joint_name": "panda_joint8",
    "ee_link_name": "panda_hand",
    "rest_arm_qpos": FRANKA_HOME_QPOS,
}

FRANKA_HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
```

### Action Space

```
Action: (1, 7) = [gripper, vx, vy, vz, drx, dry, drz]
```

| Index | Name | Description |
|-------|------|-------------|
| 0 | gripper | 0=open, 1=close |
| 1-3 | vx, vy, vz | EEF velocity in world frame (m/s equivalent) |
| 4-6 | drx, dry, drz | EEF orientation velocity (rad/s equivalent) |

### Observation Space

```
Observation: (1, 13) = [eef_xyz(3), x_dir(3), z_dir(3), gravity(3), grip(1)]
```

| Indices | Name | Description |
|---------|------|-------------|
| 0-2 | eef_xyz | End-effector position in world frame |
| 3-5 | x_dir | EEF X-axis direction (unit vector) |
| 6-8 | z_dir | EEF Z-axis direction (unit vector) |
| 9-11 | gravity | Gravity vector (always [0, 0, -1]) |
| 12 | grip | Gripper state: 1 if grasped, 0 otherwise |

---

## Franka Link Structure

```
panda_link0 (base, fixed to world)
  └── panda_link1 (joint1: revolute)
        └── panda_link2 (joint2: revolute)
              └── panda_link3 (joint3: revolute)
                    └── panda_link4 (joint4: revolute)
                          └── panda_link5 (joint5: revolute)
                                └── panda_link6 (joint6: revolute)
                                      └── panda_link7 (joint7: revolute)
                                            └── panda_link8 (joint8: fixed)
                                                  └── panda_hand (hand_joint: fixed)
                                                        ├── panda_leftfinger (finger_joint1: prismatic)
                                                        ├── panda_rightfinger (finger_joint2: prismatic)
                                                        └── panda_grasptarget (virtual link for IK)
```

### Link Positions in Home Pose

| Link | Z Position | Notes |
|------|------------|-------|
| panda_hand | 0.5503 | Gripper base |
| panda_leftfinger | 0.5119 | Finger root |
| panda_rightfinger | 0.5119 | Finger root |
| panda_grasptarget | 0.4853 | Virtual fingertip target |

### IK Behavior

**Current config uses `panda_hand`**:
- Consistent IK solutions
- ~4cm Z offset (gripper base, not fingertip)
- Commands to Z=0.03 achieve actual Z≈0.07

**Alternative `panda_grasptarget`**:
- Accurate fingertip positioning
- Causes arm self-collision during large movements
- Not recommended for current sketch-based trajectories

---

## Control Parameters

### Demo Mode (Stable)

```python
kp = 5.0    # Position gain
kd = 2.0    # Derivative gain
sub_steps = 50  # Steps per action

# Sub-step interpolation for smooth motion
for st in range(sub_steps):
    alpha = (st + 1) / sub_steps
    interp_qpos = curr_qpos + alpha * (target_qpos - curr_qpos)
    robot.move_to_qpos(interp_qpos, mode=pybullet.POSITION_CONTROL, kp=kp, kd=kd)
    sim.stepSimulation()
```

### High Gains (Unstable)

```python
# DON'T USE: causes objects to fly away
kp = 200  # Too aggressive
```

High gains cause rapid acceleration that transfers momentum to grasped objects, launching them.

### Simulation Parameters

```python
self.sim.setGravity(0, 0, -9.81)
self.sim.setTimeStep(1.0 / 240)  # 240 Hz physics
self.freq = 10  # 10 Hz control frequency
steps_per_action = 24  # 240/10 = 24 physics steps per control step
```

---

## Grasping System

### Constraint-Based Grasping

Instead of simulating finger contact, grasping creates a fixed constraint:

```python
# When gripper closes and EEF near object
constraint_id = sim.createConstraint(
    parentBodyUniqueId=robot_id,
    parentLinkIndex=hand_link_idx,
    childBodyUniqueId=object_id,
    childLinkIndex=-1,  # base link
    jointType=pybullet.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=local_offset,  # offset in gripper frame
    childFramePosition=[0, 0, 0],
)
sim.changeConstraint(constraint_id, maxForce=2000)  # High constraint force
```

### Grasp Detection Logic

```python
def _try_grasp(self):
    if self.constraint_id is not None:
        return  # Already grasping

    gripper_cmd = self.last_gripper_cmd
    if gripper_cmd < 0.9:
        return  # Gripper not commanded closed

    eef_pos = self._get_eef_pos()

    for i, obj_id in enumerate(self.rigid_ids):
        if not self._rigid_graspable[i]:
            continue

        obj_pos, _ = self.sim.getBasePositionAndOrientation(obj_id)
        dist = np.linalg.norm(np.array(eef_pos) - np.array(obj_pos))

        if dist < self.grasp_threshold:
            # Create constraint
            ...
```

### Constraint Drift Issue

Despite `maxForce=2000N`, grasped objects can drift or rotate during arm movement:
- Constraint solver has finite stiffness
- Rapid arm movements cause momentum transfer
- Sub-step interpolation helps but doesn't eliminate

---

## Mixed-Frame Sketch System

### Concept

Demo generation uses sketches with mixed coordinate frames:
- **Object-relative phases**: Rotated with the randomly-placed object
- **World-frame phases**: Fixed regardless of object rotation

This creates the symmetry conflict PEFM needs to learn.

### Implementation

```python
def split_and_rotate_sketch(sketch, object_phases, object_rotation):
    """
    Rotate object-relative phases by object_rotation.
    Leave world-frame phases unchanged.

    Args:
        sketch: List of (grip, x, y, z) tuples
        object_phases: Set of phase indices that are object-relative
        object_rotation: Angle in radians to rotate object-relative phases

    Returns:
        Rotated sketch
    """
    rotated = []
    cos_r, sin_r = np.cos(object_rotation), np.sin(object_rotation)

    for i, (grip, x, y, z) in enumerate(sketch):
        if i in object_phases:
            # Rotate around origin
            new_x = x * cos_r - y * sin_r
            new_y = x * sin_r + y * cos_r
            rotated.append((grip, new_x, new_y, z))
        else:
            # World-frame: no rotation
            rotated.append((grip, x, y, z))

    return rotated
```

### Example: peg_insert Sketch

```python
# Peg spawns at angle θ; socket at FIXED position
# Object-relative phases: approach, descend, grasp, lift
# World-frame phases: move over socket, insert

object_phases = {1, 2, 3, 4}

sketch = [
    (0, 0.35, 0.0, 0.30),   # Phase 0: World-frame - safe height
    (0, 0.0, 0.0, 0.30),    # Phase 1: Object-relative - above peg
    (0, 0.0, 0.0, 0.08),    # Phase 2: Object-relative - descend
    (1, 0.0, 0.0, 0.04),    # Phase 3: Object-relative - grasp
    (1, 0.0, 0.0, 0.25),    # Phase 4: Object-relative - lift
    (1, 0.35, -0.2, 0.25),  # Phase 5: World-frame - over socket
    (0, 0.35, -0.2, 0.02),  # Phase 6: World-frame - insert
]
```

---

## Troubleshooting Guide

### Problem: Objects fly away during manipulation

**Symptoms**: Object suddenly accelerates and exits workspace when arm moves.

**Cause**: High position control gains (kp) cause rapid joint acceleration.

**Fix**:
```python
# Use moderate gains with interpolation (demo mode)
kp = 5.0
kd = 2.0

# Interpolate between current and target joint positions
for st in range(sub_steps):
    alpha = (st + 1) / sub_steps
    interp_qpos = curr_qpos + alpha * (target_qpos - curr_qpos)
    robot.move_to_qpos(interp_qpos, kp=kp, kd=kd)
    sim.stepSimulation()
```

### Problem: Grasp fails (constraint_id is None)

**Symptoms**: Gripper closes but no constraint created. Object not attached.

**Causes**:
1. EEF not close enough to object (`dist > grasp_threshold`)
2. Object not in `rigid_graspable` list
3. Gripper command below threshold (`< 0.9`)

**Fixes**:
- Check trajectory reaches object center
- Verify `_rigid_graspable[i] = True` for target object
- Ensure sketch has `grip=1` at grasp position

### Problem: IK reaches wrong Z position

**Symptoms**: Commanded Z=0.03 but EEF achieves Z≈0.07 (or similar offset).

**Cause**: `panda_hand` link is gripper base, not fingertip. ~4cm offset.

**Workarounds**:
1. Accept offset and adjust sketch waypoints by +4cm
2. Switch to `panda_grasptarget` EE link (causes collision issues)
3. Use finger link for IK (not tested)

### Problem: Arm collides with object during approach

**Symptoms**: Object knocked away before grasp attempt.

**Cause**: Diagonal movement from home position passes through object.

**Fix**: Add safe_z waypoint for vertical approach:
```python
sketch = [
    (0, 0.35, 0.0, 0.30),   # Go to safe height first (world-frame)
    (0, obj_x, obj_y, 0.30), # Move laterally at safe height
    (0, obj_x, obj_y, 0.08), # Descend vertically
    ...
]
```

### Problem: Grasped object drifts/rotates during movement

**Symptoms**: Object position/orientation changes during arm transport.

**Cause**: Constraint solver finite stiffness, momentum transfer.

**Mitigations**:
1. Increase constraint force: `sim.changeConstraint(id, maxForce=5000)`
2. Slower movements: Reduce velocity scale in sketch interpolation
3. Smaller sub-steps: Increase `sub_steps` from 50 to 100
4. Accept drift: Design reward function with tolerance

### Problem: Real-time visualization not working

**Symptoms**: `--vis` flag doesn't show GUI window.

**Causes**:
1. Display not available (SSH without X11)
2. PyBullet built without GUI support

**Fixes**:
1. Use X11 forwarding: `ssh -X user@host`
2. Reinstall pybullet: `pip install pybullet --force-reinstall`

---

## peg_insert Task Issues

The peg insertion task currently fails. Here are the specific issues:

### Issue 1: No Wrist Rotation Control

**Problem**: Sketch has no `drz` actions to rotate the gripper.

**Context**: Peg spawns with random C4 rotation (0, 90, 180, 270 deg). Socket expects keyway facing +X. Current sketch only controls position, not orientation.

**Potential fix**: Add rotation phases to sketch:
```python
# After grasping, before insertion
(1, socket_x, socket_y, safe_z, 0, 0, -peg_rotation),  # Rotate to align
```

### Issue 2: Collision During Approach

**Problem**: Arm knocks peg before grasp.

**Context**: Peg spawn at radius 0.4m can put arm link through peg during approach.

**Potential fixes**:
1. Safe height approach (partially implemented)
2. Compute approach angle based on object position
3. Adjust spawn radius to safer zone

### Issue 3: Peg Rotates When Grasped

**Problem**: Observed ~90deg rotation when constraint created.

**Context**: Constraint frame offset computation may not preserve orientation.

**Investigation needed**:
- Check parent/child frame orientations in constraint
- Verify grasp point selection doesn't induce torque

---

## File Reference

| File | Description |
|------|-------------|
| `pefm_envs/sim_franka/franka_env.py` | Base environment class |
| `pefm_envs/sim_franka/peg_insert_env.py` | Peg insertion task (C4) |
| `pefm_envs/sim_franka/cup_pour_env.py` | Cup pouring task |
| `pefm_envs/sim_franka/book_insert_env.py` | Book to shelf task |
| `pefm_envs/sim_franka/generate_demos.py` | Demo generation script |
| `pefm_envs/sim_mobile/utils/info.py` | Robot configuration (FRANKA_HOME_QPOS) |
| `pefm_envs/sim_mobile/utils/bullet_robot.py` | BulletRobot class for IK, control |
