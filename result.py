import os
import time
import numpy as np
import usimpy
import torch
from net import IFENet
import cv2 as cv
from collections import OrderedDict
import torch.nn as nn

class IFENet(nn.Module):
    def __init__(self):
        super(IFENet, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2,padding=3),
            nn.BatchNorm2d(32),
            # nn.Dropout(self.dropout_vec[0]),
            nn.ReLU(),
            nn.Conv2d(32, 48, kernel_size=5, stride=2,padding=2),
            nn.BatchNorm2d(48),
            # nn.Dropout(self.dropout_vec[1]),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(64),
            # nn.Dropout(self.dropout_vec[2]),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(64),
            # nn.Dropout(self.dropout_vec[3]),
            nn.ReLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(96),
            # nn.Dropout(self.dropout_vec[3]),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(96),
            # nn.Dropout(self.dropout_vec[3]),
            nn.ReLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(128),
            # nn.Dropout(self.dropout_vec[4]),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm2d(128),
            # nn.Dropout(self.dropout_vec[5]),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm2d(256),
            # nn.Dropout(self.dropout_vec[6]),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=0),
            nn.BatchNorm2d(256),
            # nn.Dropout(self.dropout_vec[7]),
            nn.ReLU(),
        )

        self.img_fc = nn.Sequential(
                nn.Linear(256*4*9, 512),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.Dropout(0.3),
                nn.ReLU(),
            )

        self.speed_fc = nn.Sequential(
                nn.Linear(1, 128),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.Dropout(0.5),
                nn.ReLU(),
            )

        
        self.emb_fc = nn.Sequential(
                nn.Linear(512+128, 512),
                nn.Dropout(0.5),
                nn.ReLU(),
            )
        
        self.pred_speed_branch = nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        
        self.pred_angle_branch = nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, image, speed):
        image = (image.permute(0,3,1,2).float()-128)/128

        speed = speed.unsqueeze(1)
        image = self.conv_block(image)
        b,c,h,w = image.size()
        image = image.view(b,c*h*w)
        image = self.img_fc(image)
        speed = self.speed_fc(speed)
        embed = self.emb_fc(torch.cat([image,speed],1))
        pred_speed = self.pred_speed_branch(embed)
        pred_angle = self.pred_angle_branch(embed)
        return pred_speed.squeeze(), pred_angle.squeeze()


# connection
id = usimpy.UsimCreateApiConnection("127.0.0.1", 17771, 5000, 10000)

# start simulation
ret = usimpy.UsimStartSim(id, 10000)

## control
control = usimpy.UsimSpeedAdaptMode()
control.expected_speed = 0.0 # m/s
control.steering_angle = 0.0 # angle
control.handbrake_on = False
## action
action = usimpy.UsimVehicleState()
## collision
collision = usimpy.UsimCollision()
## image
image = usimpy.UsimCameraResponse()

model = IFENet().cuda()

new_state_dict = OrderedDict()
for k, v in torch.load("eval.pth.tar").items():
    name = k[7:] # remove module.
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)

count = 0
time1 = 0
time2 = 0
while(1):
    # control vehicle via speed & steer
    
    # get vehicle post & action
    ret = usimpy.UsimGetVehicleState(id, action)
    # get collision
    ret =usimpy.UsimGetCollisionInformation(id, collision)
    # get RGB image
    ret = usimpy.UsimGetOneCameraResponse(id, 0, image)
    # save image
    img = np.array(image.image[0:480*320*3])
    img = img.reshape((320, 480, 3))
    img = np.uint8(img[:,:,::-1])

    img = torch.from_numpy(img).cuda()
    speed =  torch.tensor(float(action.forward_speed)).float().cuda() / 50
    with torch.no_grad():
        pred_speed, pred_angle = model(img.unsqueeze(0),speed.unsqueeze(0))
        pred_speed =  pred_speed.cpu().numpy()*50
        pred_angle = pred_angle.cpu().numpy()*10
        control.expected_speed = pred_speed
        control.steering_angle = pred_angle
        control.handbrake_on = False
        print(pred_speed,pred_angle)
        ret = usimpy.UsimSetVehicleControlsBySA(id, control)
    count = count + 1
# stop simulation
ret = usimpy.UsimStopSim(id)


