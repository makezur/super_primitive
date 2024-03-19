import torch
import torch.nn as nn
import torch.nn.functional as F
import geffnet


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        basemodel_name = 'tf_efficientnet_b5_ap'
        basemodel = geffnet.create_model(basemodel_name, pretrained=True)
        # Remove last layer
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        self.original_model = basemodel

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


def norm_normalize(norm_out):
    min_kappa = 0.01
    norm_x, norm_y, norm_z, kappa = torch.split(norm_out, 1, dim=1)
    norm = torch.sqrt(norm_x ** 2.0 + norm_y ** 2.0 + norm_z ** 2.0) + 1e-10
    kappa = F.elu(kappa) + 1.0 + min_kappa
    final_out = torch.cat([norm_x / norm, norm_y / norm, norm_z / norm, kappa], dim=1)
    return final_out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # hyper-parameter for sampling
        self.sampling_ratio = 0.4
        self.importance_ratio = 0.7

        # feature-map
        self.conv2 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0)
        self.up1 = UpSampleBN(skip_input=2048 + 176, output_features=1024)
        self.up2 = UpSampleBN(skip_input=1024 + 64, output_features=512)
        self.up3 = UpSampleBN(skip_input=512 + 40, output_features=256)
        self.up4 = UpSampleBN(skip_input=256 + 24, output_features=128)

        # produces 1/8 res output
        self.out_conv_res8 = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)

        # produces 1/4 res output
        self.out_conv_res4 = nn.Sequential(
            nn.Conv1d(512 + 4, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 4, kernel_size=1),
        )

        # produces 1/2 res output
        self.out_conv_res2 = nn.Sequential(
            nn.Conv1d(256 + 4, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 4, kernel_size=1),
        )

        # produces 1/1 res output
        self.out_conv_res1 = nn.Sequential(
            nn.Conv1d(128 + 4, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1), nn.ReLU(),
            nn.Conv1d(128, 4, kernel_size=1),
        )

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]

        # generate feature-map
        x_d0 = self.conv2(x_block4)                     # x_d0 : [2, 2048, 15, 20]      1/32 res
        x_d1 = self.up1(x_d0, x_block3)                 # x_d1 : [2, 1024, 30, 40]      1/16 res
        x_d2 = self.up2(x_d1, x_block2)                 # x_d2 : [2, 512, 60, 80]       1/8 res
        x_d3 = self.up3(x_d2, x_block1)                 # x_d3: [2, 256, 120, 160]      1/4 res
        x_d4 = self.up4(x_d3, x_block0)                 # x_d4: [2, 128, 240, 320]      1/2 res

        # 1/8 res output
        out_res8 = self.out_conv_res8(x_d2)             # out_res8: [2, 4, 60, 80]      1/8 res output
        out_res8 = norm_normalize(out_res8)             # out_res8: [2, 4, 60, 80]      1/8 res output

        # out_res4
        feat_map = F.interpolate(x_d2, scale_factor=2, mode='bilinear', align_corners=True)
        init_pred = F.interpolate(out_res8, scale_factor=2, mode='bilinear', align_corners=True)
        feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 512+4, H, W)
        B, _, H, W = feat_map.shape

        out_res4 = self.out_conv_res4(feat_map.view(B, 512 + 4, -1))  # (B, 4, N)
        out_res4 = norm_normalize(out_res4)  # (B, 4, N) - normalized
        out_res4 = out_res4.view(B, 4, H, W)

        # out_res2
        feat_map = F.interpolate(x_d3, scale_factor=2, mode='bilinear', align_corners=True)
        init_pred = F.interpolate(out_res4, scale_factor=2, mode='bilinear', align_corners=True)
        feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 512+4, H, W)
        B, _, H, W = feat_map.shape

        out_res2 = self.out_conv_res2(feat_map.view(B, 256 + 4, -1))  # (B, 4, N)
        out_res2 = norm_normalize(out_res2)  # (B, 4, N) - normalized
        out_res2 = out_res2.view(B, 4, H, W)

        # out_res1
        feat_map = F.interpolate(x_d4, scale_factor=2, mode='bilinear', align_corners=True)
        init_pred = F.interpolate(out_res2, scale_factor=2, mode='bilinear', align_corners=True)
        feat_map = torch.cat([feat_map, init_pred], dim=1)  # (B, 512+4, H, W)
        B, _, H, W = feat_map.shape

        out_res1 = self.out_conv_res1(feat_map.view(B, 128 + 4, -1))  # (B, 4, N)
        out_res1 = norm_normalize(out_res1)  # (B, 4, N) - normalized
        out_res1 = out_res1.view(B, 4, H, W)

        return [out_res1], None, None


class NNET(nn.Module):
    def __init__(self, args=None):
        super(NNET, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()


    def forward(self, img, **kwargs):
        return self.decoder(self.encoder(img))
    


# load model
def load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location='cpu')['model']

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    return model



if __name__ == '__main__':
    checkpoint = './checkpoints/scannet.pt'
    model = NNET().to(0)
    model = load_checkpoint(checkpoint, model)
    model.eval()

    x = torch.rand(2, 3, 480, 640)
    x = x.to(0)

    norm_out = model(x)
    print(norm_out.shape)
