
# Actual Test
import cwiid
import time
import numpy as np
import time
import matplotlib.pyplot as plt
from datetime import datetime
# connecting to the Wiimote. This allows several attempts
# as first few often fail.


CALIBRATION_TRIES = 100
WAIT_TIME = 0.01

ground_amount = None
wm = None

to_log = []

def log(*x):
    global to_log
    print x
    to_log.append(" ".join([str(s) for s in x]))
	

def load_wii():
    global wm

    log("Press 1+2 on your Wiimote now...")
    
    attempt_i = 2
    while not wm:
        try:
            wm = cwiid.Wiimote()
        except RuntimeError:
            if (attempt_i > 10):
                quit()
                break
            log("Error opening wiimote connection")
            log("attempt " + str(attempt_i))
            attempt_i += 1

    log("Loaded Wiimote")

    # set Wiimote to report button presses and accelerometer state
    wm.rpt_mode = cwiid.RPT_ACC

    # turn on led to show connected
    wm.led = 15

load_wii()

take_values = 100

average_ground = None


def write_log():
    global to_log
    with open("saved-data-"+datetime.now().strftime("%m-%d-%Y-%H:%M")+".txt", "w+") as f:
        for line in to_log:
            f.write(line+"\n")
        to_log = []

def do_select():
    global take_values
    raw_input("Press enter to calibrate")
    do_calibrate()
    take_values = int(raw_input("How many values?\n:"))
    raw_input("Press enter to run")
    do_run()


def do_calibrate():
    global base_level,ground_amount
    
    log("Calibration data")
    calibration_states = []
    for i in range(CALIBRATION_TRIES):
        cur_acc =wm.state["acc"]
        calibration_states.append(cur_acc)
        log(cur_acc)
        time.sleep(WAIT_TIME)
        

    base_level = np.average(calibration_states, axis=0)
    ground_amount = np.max(base_level) - np.min(base_level)
    
    ground_amount_i = np.argmax(base_level)
    base_level[ground_amount_i] -= ground_amount
    log("Base level:", base_level)
    log("g amount:", ground_amount)


def get_time():
    return time.time()

def do_run():
    start_time = get_time()
    xs = []
    ys = []
    for val_i in range(take_values):
        t = get_time()
        cur = np.array(wm.state['acc'])
        #print "Current value", cur
        log(t, cur)
        xs.append(t)
        ys.append(cur)
        time.sleep(0.0005)
    
    for dim in range(len(ys[0])):
        dim_y = [y[dim] for y in ys]
        plt.plot(xs, dim_y)
    plt.show()


log("Calibrate")

#write_log()

try:
    do_select()
    
except Exception as e:
	print e
finally:
    write_log()
    wm.close()    

