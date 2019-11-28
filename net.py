import torch
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

# if __name__ == "__main__":
#     model = IFENet()
#     img = torch.randn((4,3,320,480))
#     speed = torch.randn((4,1))
#     pred_speed,pred_angle = model(img,speed)

#     print(pred_angle.size())