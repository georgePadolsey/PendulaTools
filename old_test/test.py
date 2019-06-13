import cwiid
import time
import i2c
import numpy as np
from time import sleep
# connecting to the Wiimote. This allows several attempts
# as first few often fail.


CALIBRATION_TRIES = 100
WAIT_TIME = 0.01

ground_amount = None

print 'Press 1+2 on your Wiimote now...'
wm = None
attempt_i = 2
while not wm:
    try:
        wm = cwiid.Wiimote()
    except RuntimeError:
        if (attempt_i > 10):
            quit()
            break
        print "Error opening wiimote connection"
        print "attempt " + str(i)
        attempt_i += 1
        sleep(.5)

# set Wiimote to report button presses and accelerometer state
wm.rpt_mode = cwiid.RPT_BTN | cwiid.RPT_ACC

# turn on led to show connected
wm.led = 15


average_ground = None


def do_select():
    #
    do_calibrate()
    do_run()


def do_calibrate():
    global base_level

    calibration_states = []
    for i in range(CALIBRATION_TRIES):
        calibration_states.append(wm.state["acc"])
        sleep(WAIT_TIME)

    base_level = np.average(calibration_states, axis=0)
    ground_amount = np.max(base_level) - np.min(base_level)

    print "Base level:", base_level
    print "g amount:", ground_amount


def do_run():
    while True:
        cur = np.array(wm.state['acc']) - np.array(base_level)
        print "Current value", cur/ground_amount
        time.sleep(0.01)


print("Calibrate")

running = True
while running:
    do_select()
