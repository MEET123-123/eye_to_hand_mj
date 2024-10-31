from sksurgerynditracker.nditracker import NDITracker

# TODO:修改ROM文件
SETTINGS = {
    "tracker type": "polaris",
    #         "romfiles": ["./NDI_Rom/8700339qjx.rom"]
    "romfiles": ["E:/eye_to_hand/tip.rom"]
}
TRACKER = NDITracker(SETTINGS)
TRACKER.start_tracking()

for i in range(100):
    port_handles, timestamps, framenumbers, tracking, quality = TRACKER.get_frame()
    ret = tracking[0]
    print(ret)

TRACKER.stop_tracking()
TRACKER.close()