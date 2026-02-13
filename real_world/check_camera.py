import pyrealsense2 as rs

ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("No RealSense devices found. (Check cable/permissions)")
else:
    print(f"Found {len(devices)} device(s):")
    for dev in devices:
        print(f" - Name: {dev.get_info(rs.camera_info.name)}")
        print(f" - Serial: {dev.get_info(rs.camera_info.serial_number)}")
        print(f" - Firmware: {dev.get_info(rs.camera_info.firmware_version)}")