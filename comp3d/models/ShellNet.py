"""
https://github.com/Dung-Han-Lee/Pointcloud-based-Row-Detection-using-ShellNet-and-PyTorch/
"""

import torch.nn as nn
import torch


def knn(points, queries, K):
    """
    Args:
        points   ( B x N x 3 tensor )
        queries  ( B x M x 3 tensor )  M < N
        K        (constant) num of neighbors
    Outputs:
        knn     (B x M x K x 3 tensor) sorted K nearest neighbor
        indices (B x M x K tensor) knn indices
    """
    value = None
    indices = None
    num_batch = points.shape[0]
    for i in range(num_batch):
        point = points[i]
        query = queries[i]
        dist = torch.cdist(point, query)
        idxs = dist.topk(K, dim=0, largest=False, sorted=True).indices
        idxs = idxs.transpose(0, 1)
        nn = point[idxs].unsqueeze(0)
        value = nn if value is None else torch.cat((value, nn))

        idxs = idxs.unsqueeze(0)
        indices = idxs if indices is None else torch.cat((indices, idxs))

    # why long???
    # return value.long(), indices.long()
    return value.float(), indices.long()


def gather_feature(features, indices):
    """
    Args:
        features ( B x N x F tensor) -- feature from previous layer
        indices  ( B x M x K tensor) --  represents queries' k nearest neighbor
    Output:
        features ( B x M x K x F tensor) -- knn features from previous layer 
    """
    res = None
    num_batch = features.shape[0]
    for B in range(num_batch):
        knn_features = features[B][indices[B]].unsqueeze(0)
        res = knn_features if res is None else torch.cat((res, knn_features))
    return res


def random_sample(points, num_sample):
    """
    Args:
        points ( B x N x 3 tensor )
        num_sample (constant)
    Outputs:
        sampled_points (B x num_sample x 3 tensor)
    """
    perm = torch.randperm(points.shape[1])
    return points[:, perm[:num_sample]].clone()


class Dense(nn.Module):
    def __init__(self, in_size, out_size, in_dim=3,
                 has_bn=True, drop_out=None):
        super(Dense, self).__init__()
        """
        Args:
            input ( B x M x K x 3  tensor ) -- subtraction vectors 
                from query to its k nearest neighbor
        Output: 
            local point feature ( B x M x K x 64 tensor ) 
        """
        self.has_bn = has_bn
        self.in_dim = in_dim

        if in_dim == 3:
            self.batchnorm = nn.BatchNorm1d(in_size)
        elif in_dim == 4:
            self.batchnorm = nn.BatchNorm2d(in_size)
        else:
            self.batchnorm = None

        if drop_out is None:
            self.linear = nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.ReLU()
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.ReLU(),
                nn.Dropout(drop_out)
            )

    def forward(self, inputs):

        if self.has_bn == True:
            d = self.in_dim - 1
            outputs = self.batchnorm(inputs.transpose(1, d)).transpose(1, d)
            outputs = self.linear(outputs)
            return outputs

        else:
            outputs = self.linear(inputs)
            return outputs


class ShellConv(nn.Module):
    def __init__(self, out_features, prev_features, neighbor, division,
                 has_bn=True):
        super(ShellConv, self).__init__()
        """
        out_features  (int) num of output feature (dim = -1)
        prev_features (int) num of prev feature (dim = -1)
        neighbor      (int) num of nearest neighbor in knn
        division      (int) num of division
        """

        self.K = neighbor
        self.S = int(self.K / division)  # num of feaure per shell
        self.F = 64  # num of local point features
        self.neighbor = neighbor
        in_channel = self.F + prev_features
        out_channel = out_features

        self.dense1 = Dense(3, self.F // 2, in_dim=4, has_bn=has_bn)
        self.dense2 = Dense(self.F // 2, self.F, in_dim=4, has_bn=has_bn)
        self.maxpool = nn.MaxPool2d((1, self.S), stride=(1, self.S))
        if has_bn == True:
            self.conv = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                nn.Conv2d(in_channel, out_channel, (1, division)),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, (1, division)),
                nn.ReLU(),
            )

    def forward(self, points, queries, prev_features):
        """
        Args:
            points          (B x N x 3 tensor)
            query           (B x M x 3 tensor) -- note that M < N
            prev_features   (B x N x F1 tensor)
        Outputs:
            feat            (B x M x F2 tensor)
        """

        knn_pts, idxs = knn(points, queries, self.K)
        knn_center = queries.unsqueeze(2)
        knn_points_local = knn_center - knn_pts

        knn_feat_local = self.dense1(knn_points_local)
        knn_feat_local = self.dense2(knn_feat_local)

        # shape: B x M x K x F
        if prev_features is not None:
            knn_feat_prev = gather_feature(prev_features, idxs)
            knn_feat_cat = torch.cat((knn_feat_local, knn_feat_prev), dim=-1)
        else:
            knn_feat_cat = knn_feat_local

        knn_feat_cat = knn_feat_cat.permute(0, 3, 1, 2)  # BMKF -> BFMK
        knn_feat_max = self.maxpool(knn_feat_cat)
        output = self.conv(knn_feat_max).permute(0, 2, 3, 1)

        return output.squeeze(2)


class ShellConv_RI(nn.Module):
    def __init__(self, out_features, prev_features, neighbor, division,
                 has_bn=True):
        super(ShellConv_RI, self).__init__()
        """
        out_features  (int) num of output feature (dim = -1)
        prev_features (int) num of prev feature (dim = -1)
        neighbor      (int) num of nearest neighbor in knn
        division      (int) num of division
        """

        self.K = neighbor
        self.S = int(self.K / division)  # num of feaure per shell
        self.F = 64  # num of local point features
        self.neighbor = neighbor
        in_channel = self.F + prev_features
        out_channel = out_features

        self.dense1 = Dense(6, self.F // 2, in_dim=4, has_bn=has_bn)
        self.dense2 = Dense(self.F // 2, self.F, in_dim=4, has_bn=has_bn)
        self.maxpool = nn.MaxPool2d((1, self.S), stride=(1, self.S))
        if has_bn == True:
            self.conv = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                nn.Conv2d(in_channel, out_channel, (1, division)),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, (1, division)),
                nn.ReLU(),
            )

    def forward(self, points, queries, prev_features):
        """
        Args:
            points          (B x N x 3 tensor)
            query           (B x M x 3 tensor) -- note that M < N
            prev_features   (B x N x F1 tensor)
        Outputs:
            feat            (B x M x F2 tensor)
        """

        B, N, _ = points.size()

        knn_pts, idxs = knn(points, queries, self.K)
        knn_center = queries.unsqueeze(2)
        knn_points_local = knn_center - knn_pts

        knn_B, knn_M, knn_K, _ = knn_pts.size()

        # calc (global centroid) - (knn_points)
        centroid = torch.sum(points, dim=1) / N
        # print (centroid.size())
        centroid_reshape = centroid.unsqueeze(1).expand(-1, knn_M * knn_K, -1)
        # print (centroid_reshape.size())
        centroid_reshape = torch.reshape(centroid_reshape, (knn_B, knn_M, knn_K, 3))
        # print (centroid_reshape.size())
        knn_dir_global = centroid_reshape - knn_pts
        # print (knn_dir_global.size())
        # print ('\n')

        # calc (local centroid) - (knn_points_local)
        knn_centroid_local = torch.sum(knn_points_local, dim=2) / knn_K
        # print (knn_centroid_local.size())
        knn_centroid_reshape = knn_centroid_local.unsqueeze(2).expand(-1, -1, knn_K, -1)
        # print (knn_centroid_reshape.size())
        knn_dir_local = knn_centroid_reshape - knn_points_local
        # print (knn_dir_local.size())
        # print ('\n')

        inputs = torch.cat((knn_dir_local, knn_dir_global), dim=-1)

        knn_feat_local = self.dense1(inputs)
        knn_feat_local = self.dense2(knn_feat_local)

        # shape: B x M x K x F
        if prev_features is not None:
            knn_feat_prev = gather_feature(prev_features, idxs)
            knn_feat_cat = torch.cat((knn_feat_local, knn_feat_prev), dim=-1)
        else:
            knn_feat_cat = knn_feat_local

        knn_feat_cat = knn_feat_cat.permute(0, 3, 1, 2)  # BMKF -> BFMK
        knn_feat_max = self.maxpool(knn_feat_cat)
        output = self.conv(knn_feat_max).permute(0, 2, 3, 1)

        return output.squeeze(2)


class ShellUp(nn.Module):
    def __init__(self, out_features, prev_features, neighbor, division, has_bn=True):
        super(ShellUp, self).__init__()
        self.has_bn = has_bn
        self.sconv = ShellConv(out_features, prev_features, neighbor,
                               division, has_bn)
        self.dense = Dense(2 * out_features, out_features, has_bn=has_bn)

    def forward(self, points, queries, prev_features, feat_skip_connect):
        sconv = self.sconv(points, queries, prev_features)
        feat_cat = torch.cat((sconv, feat_skip_connect), dim=-1)

        outputs = self.dense(feat_cat)
        return outputs


class ShellUp_RI(nn.Module):
    def __init__(self, out_features, prev_features, neighbor, division, has_bn=True):
        super(ShellUp_RI, self).__init__()
        self.has_bn = has_bn
        self.sconv = ShellConv_RI(out_features, prev_features, neighbor, division, has_bn)
        self.dense = Dense(2 * out_features, out_features, has_bn=has_bn)

    def forward(self, points, queries, prev_features, feat_skip_connect):
        sconv = self.sconv(points, queries, prev_features)
        feat_cat = torch.cat((sconv, feat_skip_connect), dim=-1)

        outputs = self.dense(feat_cat)
        return outputs


class ShellNet_Feature(nn.Module):
    def __init__(self, num_points, out_dim=1024, conv_scale=1, dense_scale=1, has_bn=True, global_feat=True):
        super(ShellNet_Feature, self).__init__()
        self.num_points = num_points
        self.global_feat = global_feat
        filters = [64, 128, 256, 512]
        filters = [int(x / conv_scale) for x in filters]

        features = [256, 128]
        features = [int(x / dense_scale) for x in features]

        self.shellconv1 = ShellConv(filters[1], 0, 32, 4, has_bn)
        self.shellconv2 = ShellConv(filters[2], filters[1], 16, 2, has_bn)
        self.shellconv3 = ShellConv(filters[3], filters[2], 8, 1, has_bn)

        self.fc1 = Dense(filters[3], features[0], has_bn=has_bn, drop_out=0)
        self.fc2 = Dense(features[0], features[1], has_bn=has_bn, drop_out=0.5)
        self.fc3 = Dense(features[1], out_dim, has_bn=has_bn)

    def forward(self, inputs):
        query1 = random_sample(inputs, int(self.num_points // 2))
        sconv1 = self.shellconv1(inputs, query1, None)
        # print("sconv1.shape = ", sconv1.shape)

        query2 = random_sample(query1, int(self.num_points // 4))
        sconv2 = self.shellconv2(query1, query2, sconv1)
        # print("sconv2.shape = ", sconv2.shape)

        query3 = random_sample(query2, int(self.num_points // 8))
        sconv3 = self.shellconv3(query2, query3, sconv2)
        # print("sconv3.shape = ", sconv3.shape)

        fc1 = self.fc1(sconv3)
        # print("fc1.shape = ", fc1.shape)

        fc2 = self.fc2(fc1)
        # print("fc2.shape = ", fc2.shape)

        output = self.fc3(fc2)
        # print("fc3.shape = ", output.shape)

        if self.global_feat:
            output, _ = torch.max(output, 1)
            output = output.view(-1, 1024)

        return output


class ShellNet_RI_Feature(nn.Module):
    def __init__(self, num_points, out_dim=1024, conv_scale=1, dense_scale=1, has_bn=True, global_feat=True):
        super(ShellNet_RI_Feature, self).__init__()
        self.num_points = num_points
        self.global_feat = global_feat
        filters = [64, 128, 256, 512]
        filters = [int(x / conv_scale) for x in filters]

        features = [256, 128]
        features = [int(x / dense_scale) for x in features]

        self.shellconv1 = ShellConv_RI(filters[1], 0, 32, 4, has_bn)
        self.shellconv2 = ShellConv_RI(filters[2], filters[1], 16, 2, has_bn)
        self.shellconv3 = ShellConv_RI(filters[3], filters[2], 8, 1, has_bn)

        self.fc1 = Dense(filters[3], features[0], has_bn=has_bn, drop_out=0)
        self.fc2 = Dense(features[0], features[1], has_bn=has_bn, drop_out=0.5)
        self.fc3 = Dense(features[1], out_dim, has_bn=has_bn)

    def forward(self, inputs):
        query1 = random_sample(inputs, int(self.num_points // 2))
        sconv1 = self.shellconv1(inputs, query1, None)
        # print("sconv1.shape = ", sconv1.shape)

        query2 = random_sample(query1, int(self.num_points // 4))
        sconv2 = self.shellconv2(query1, query2, sconv1)
        # print("sconv2.shape = ", sconv2.shape)

        query3 = random_sample(query2, int(self.num_points // 8))
        sconv3 = self.shellconv3(query2, query3, sconv2)
        # print("sconv3.shape = ", sconv3.shape)

        fc1 = self.fc1(sconv3)
        # print("fc1.shape = ", fc1.shape)

        fc2 = self.fc2(fc1)
        # print("fc2.shape = ", fc2.shape)

        output = self.fc3(fc2)
        # print("fc3.shape = ", output.shape)

        if self.global_feat:
            output, _ = torch.max(output, 1)
            output = output.view(-1, 1024)

        return output


class ShellNet_Seg(nn.Module):
    def __init__(self, num_class, num_points, conv_scale=1, dense_scale=1, has_bn=True):
        super(ShellNet_Seg, self).__init__()
        self.num_points = num_points
        filters = [64, 128, 256, 512]
        filters = [int(x / conv_scale) for x in filters]

        features = [256, 128]
        features = [int(x / dense_scale) for x in features]

        self.shellconv1 = ShellConv(filters[1], 0, 32, 4, has_bn)
        self.shellconv2 = ShellConv(filters[2], filters[1], 16, 2, has_bn)
        self.shellconv3 = ShellConv(filters[3], filters[2], 8, 1, has_bn)
        self.shellup3 = ShellUp(filters[2], filters[3], 8, 1, has_bn)
        self.shellup2 = ShellUp(filters[1], filters[2], 16, 2, has_bn)
        self.shellup1 = ShellConv(filters[0], filters[1], 32, 4, has_bn)

        self.fc1 = Dense(filters[0], features[0], has_bn=has_bn, drop_out=0)
        self.fc2 = Dense(features[0], features[1], has_bn=has_bn, drop_out=0.5)
        self.fc3 = Dense(features[1], num_class, has_bn=has_bn)

    def forward(self, inputs):
        query1 = random_sample(inputs, int(self.num_points // 2))
        sconv1 = self.shellconv1(inputs, query1, None)
        # print("sconv1.shape = ", sconv1.shape)

        query2 = random_sample(query1, int(self.num_points // 4))
        sconv2 = self.shellconv2(query1, query2, sconv1)
        # print("sconv2.shape = ", sconv2.shape)

        query3 = random_sample(query2, int(self.num_points // 8))
        sconv3 = self.shellconv3(query2, query3, sconv2)
        # print("sconv3.shape = ", sconv3.shape)

        up3 = self.shellup3(query3, query2, sconv3, sconv2)
        # print("up3.shape = ", up3.shape)

        up2 = self.shellup2(query2, query1, up3, sconv1)
        # print("up2.shape = ", up2.shape)

        up1 = self.shellup1(query1, inputs, up2)
        # print("up1.shape = ", up1.shape)

        fc1 = self.fc1(up1)
        # print("fc1.shape = ", fc1.shape)

        fc2 = self.fc2(fc1)
        # print("fc2.shape = ", fc2.shape)

        output = self.fc3(fc2)
        # print("fc3.shape = ", output.shape)

        return output


class ShellNet_RI_Seg(nn.Module):
    def __init__(self, num_class, num_points, conv_scale=1, dense_scale=1, has_bn=True):
        super(ShellNet_RI_Seg, self).__init__()
        self.num_points = num_points
        filters = [64, 128, 256, 512]
        filters = [int(x / conv_scale) for x in filters]

        features = [256, 128]
        features = [int(x / dense_scale) for x in features]

        self.shellconv1 = ShellConv_RI(filters[1], 0, 32, 4, has_bn)
        self.shellconv2 = ShellConv_RI(filters[2], filters[1], 16, 2, has_bn)
        self.shellconv3 = ShellConv_RI(filters[3], filters[2], 8, 1, has_bn)
        self.shellup3 = ShellUp_RI(filters[2], filters[3], 8, 1, has_bn)
        self.shellup2 = ShellUp_RI(filters[1], filters[2], 16, 2, has_bn)
        self.shellup1 = ShellConv_RI(filters[0], filters[1], 32, 4, has_bn)

        self.fc1 = Dense(filters[0], features[0], has_bn=has_bn, drop_out=0)
        self.fc2 = Dense(features[0], features[1], has_bn=has_bn, drop_out=0.5)
        self.fc3 = Dense(features[1], num_class, has_bn=has_bn)

    def forward(self, inputs):
        query1 = random_sample(inputs, int(self.num_points // 2))
        sconv1 = self.shellconv1(inputs, query1, None)
        # print("sconv1.shape = ", sconv1.shape)

        query2 = random_sample(query1, int(self.num_points // 4))
        sconv2 = self.shellconv2(query1, query2, sconv1)
        # print("sconv2.shape = ", sconv2.shape)

        query3 = random_sample(query2, int(self.num_points // 8))
        sconv3 = self.shellconv3(query2, query3, sconv2)
        # print("sconv3.shape = ", sconv3.shape)

        up3 = self.shellup3(query3, query2, sconv3, sconv2)
        # print("up3.shape = ", up3.shape)

        up2 = self.shellup2(query2, query1, up3, sconv1)
        # print("up2.shape = ", up2.shape)

        up1 = self.shellup1(query1, inputs, up2)
        # print("up1.shape = ", up1.shape)

        fc1 = self.fc1(up1)
        # print("fc1.shape = ", fc1.shape)

        fc2 = self.fc2(fc1)
        # print("fc2.shape = ", fc2.shape)

        output = self.fc3(fc2)
        # print("fc3.shape = ", output.shape)

        return output


if __name__ == '__main__':
    B, M, K = 2, 1024, 32
    # Create random Tensors to hold inputs and outputs
    p = torch.randn(B, M, 3)
    q = torch.randn(B, M // 2, 3)
    f = torch.randn(B, M, 128)
    y = torch.randn(B, M // 2, 128)

    nn_pts, idxs = knn(p, q, 32)
    nn_center = q.unsqueeze(2)
    nn_points_local = nn_center - nn_pts

    model = ShellNet_Seg(2, 1024, conv_scale=1, dense_scale=1)
    print(model(p).shape)
