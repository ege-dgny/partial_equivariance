import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
import shutil

# def fix_permissions(path):
#     """Changes ownership of a file/folder from root to the actual user."""
#     # specific to running with sudo
#     sudo_uid = os.environ.get('SUDO_UID')
#     sudo_gid = os.environ.get('SUDO_GID')
    
#     if sudo_uid is not None and sudo_gid is not None:
#         os.chown(path, int(sudo_uid), int(sudo_gid))

def get_RGBD_image(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    ctx = rs.context()
    if len(ctx.devices) > 0:
        for dev in ctx.devices:
            print(f"Resetting device: {dev.get_info(rs.camera_info.name)}")
            dev.hardware_reset()
    else:
        print("No device found to reset.")
    
    print("Waiting for camera to reboot...")
    time.sleep(10) 

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)

    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            timestamp = time.time()

            depth_filename = os.path.join(output_dir, f"depth_{timestamp:.6f}.png")
            color_filename = os.path.join(output_dir, f"color_{timestamp:.6f}.png")

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imwrite(depth_filename, depth_colormap)
            # fix_permissions(color_filename)
            cv2.imwrite(color_filename, color_image)
            # fix_permissions(depth_filename)

            cv2.imshow('Depth', depth_colormap)
            cv2.imshow('Color', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception:
        print("RuntimeError: Frame didn't arrive within 5000")            
    finally:
        pipeline.stop()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    output_dir = "ping_pong_ball_free_fall"
    get_RGBD_image(output_dir=output_dir)