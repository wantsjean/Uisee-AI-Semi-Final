import os
import time
from PIL import Image
import numpy as np
import usimpy

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# connection
id = usimpy.UsimCreateApiConnection("127.0.0.1", 17771, 5000, 10000)

# start simulation
ret = usimpy.UsimStartSim(id, 10000)
print (ret)

## control
control = usimpy.UsimSpeedAdaptMode()
control.expected_speed = 0.8 # m/s
control.steering_angle = 0.0 # angle
control.handbrake_on = False
## action
action = usimpy.UsimVehicleState()
## collision
collision = usimpy.UsimCollision()
## image
image = usimpy.UsimCameraResponse()

count = 0
time1 = 0
time2 = 0
root_dir = './dataset/'
img_dir = root_dir + 'imgs/'
mkdir(img_dir)

while(1):
    # control vehicle via speed & steer
    ret = usimpy.UsimSetVehicleControlsBySA(id, control)
    # get vehicle post & action
    ret = usimpy.UsimGetVehicleState(id, action)
    # get collision
    ret =usimpy.UsimGetCollisionInformation(id, collision)
    # get RGB image
    ret = usimpy.UsimGetOneCameraResponse(id, 0, image)
    # save image
    img = np.array(image.image[0:480*320*3])
    img = img.reshape((320, 480, 3))
    img_PIL = Image.fromarray(np.uint8(img))
    img_PIL.save(img_dir+str(action.time_stamp)+".png", 'png')
    # save pose & action
    with open(root_dir + 'labels.txt', 'a+') as f:
        f.write('%d %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n' % (count, action.time_stamp, action.pose.position.x_val, \
                 action.pose.position.y_val, action.pose.position.z_val, action.pose.rotation.x_val, action.pose.rotation.y_val, \
                 action.pose.rotation.z_val, action.steering_angle, action.forward_speed))
    # auxiliary info
    time1 = action.time_stamp
    print ('time: %d, steer: %.6f, speed: %.6f, time_gap: %d, collision: %d, collision_time: %d' % (
            action.time_stamp, action.steering_angle, action.forward_speed, (time1-time2), collision.is_collided, collision.time_stamp))
    time2 = time1
    count = count + 1
# stop simulation
ret = usimpy.UsimStopSim(id)
print (ret)

