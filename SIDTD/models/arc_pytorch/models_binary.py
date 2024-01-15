import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import math

use_cuda = True


class GlimpseWindow:
    """
    Generates glimpses from images using Cauchy kernels.

    Args:
        glimpse_h (int): The height of the glimpses to be generated.
        glimpse_w (int): The width of the glimpses to be generated.

    """

    def __init__(self, glimpse_h: int, glimpse_w: int):
        self.glimpse_h = glimpse_h
        self.glimpse_w = glimpse_w

    @staticmethod
    def _get_filterbanks(delta_caps: Variable, center_caps: Variable, image_size: int, glimpse_size: int) -> Variable:
        """
        Generates Cauchy Filter Banks along a dimension.

        Args:
            delta_caps (B,):  A batch of deltas [-1, 1]
            center_caps (B,): A batch of [-1, 1] reals that dictate the location of center of cauchy kernel glimpse.
            image_size (int): size of images along that dimension
            glimpse_size (int): size of glimpses to be generated along that dimension

        Returns:
            (B, image_size, glimpse_size): A batch of filter banks

        """

        # convert dimension sizes to float. lots of math ahead.
        image_size = float(image_size)
        glimpse_size = float(glimpse_size)

        # scale the centers and the deltas to map to the actual size of given image.
        centers = (image_size - 1) * (center_caps + 1) / 2.0  # (B)
        deltas = (float(image_size) / glimpse_size) * (1.0 - torch.abs(delta_caps))

        # calculate gamma for cauchy kernel
        gammas = torch.exp(1.0 - 2 * torch.abs(delta_caps))  # (B)

        # coordinate of pixels on the glimpse
        glimpse_pixels = Variable(torch.arange(0, glimpse_size) - (glimpse_size - 1.0) / 2.0)  # (glimpse_size)
        if use_cuda:
            glimpse_pixels = glimpse_pixels.cuda()

        # space out with delta
        glimpse_pixels = deltas[:, None] * glimpse_pixels[None, :]  # (B, glimpse_size)
        # center around the centers
        glimpse_pixels = centers[:, None] + glimpse_pixels  # (B, glimpse_size)

        # coordinates of pixels on the image
        image_pixels = Variable(torch.arange(0, image_size))  # (image_size)
        if use_cuda:
            image_pixels = image_pixels.cuda()

        fx = image_pixels - glimpse_pixels[:, :, None]  # (B, glimpse_size, image_size)
        fx = fx / gammas[:, None, None]
        fx = fx ** 2.0
        fx = 1.0 + fx
        fx = math.pi * gammas[:, None, None] * fx
        fx = 1.0 / fx
        fx = fx / (torch.sum(fx, dim=2) + 1e-4)[:, :, None]  # we add a small constant in the denominator division by 0.

        return fx.transpose(1, 2)

    def get_attention_mask(self, glimpse_params: Variable, mask_h: int, mask_w: int) -> Variable:
        """
        For visualization, generate a heat map (or mask) of which pixels got the most "attention".

        Args:
            glimpse_params (B, hx):  A batch of glimpse parameters.
            mask_h (int): The height of the image for which the mask is being generated.
            mask_w (int): The width of the image for which the mask is being generated.

        Returns:
            (B, mask_h, mask_w): A batch of masks with attended pixels weighted more.

        """

        batch_size, _ = glimpse_params.size()

        # (B, image_h, glimpse_h)
        F_h = self._get_filterbanks(delta_caps=glimpse_params[:, 2], center_caps=glimpse_params[:, 0],
                                    image_size=mask_h, glimpse_size=self.glimpse_h)

        # (B, image_w, glimpse_w)
        F_w = self._get_filterbanks(delta_caps=glimpse_params[:, 2], center_caps=glimpse_params[:, 1],
                                    image_size=mask_w, glimpse_size=self.glimpse_w)

        # (B, glimpse_h, glimpse_w)
        glimpse_proxy = Variable(torch.ones(batch_size, self.glimpse_h, self.glimpse_w))

        # find the attention mask that lead to the glimpse.
        mask = glimpse_proxy
        mask = torch.bmm(F_h, mask)
        mask = torch.bmm(mask, F_w.transpose(1, 2))

        # scale to between 0 and 1.0
        mask = mask - mask.min()
        mask = mask / mask.max()
        mask = mask.float()

        return mask

    def get_glimpse(self, images: Variable, glimpse_params: Variable) -> Variable:
        """
        *Adaptation from https://github.com/gitabcworld/ConvArc to accept coulor images
        
        Generate glimpses given images and glimpse parameters. This is the main method of this class.

        The glimpse parameters are (h_center, w_center, delta). (h_center, w_center)
        represents the relative position of the center of the glimpse on the image. delta determines
        the zoom factor of the glimpse.

        Args:
            images (B, c, h, w):  A batch of images
            glimpse_params (B, 3):  A batch of glimpse parameters (h_center, w_center, delta)

        Returns:
            (B, glimpse_h, glimpse_w): A batch of glimpses.

        """
        if len(images.size())==4:
            #channels, batch_size, image_h, image_w = images.size()
            batch_size, channels, image_h, image_w = images.size()
        else:
            batch_size, image_h, image_w = images.size()

        # (B, image_h, glimpse_h)
        F_h = self._get_filterbanks(delta_caps=glimpse_params[:, 2], center_caps=glimpse_params[:, 0],
                                    image_size=image_h, glimpse_size=self.glimpse_h)

        # (B, image_w, glimpse_w)
        F_w = self._get_filterbanks(delta_caps=glimpse_params[:, 2], center_caps=glimpse_params[:, 1],
                                    image_size=image_w, glimpse_size=self.glimpse_w)

        # F_h.T * images * F_w
        glimpses = images
        # support for 1-3 channel images.
        if len(glimpses.shape)==4:
            F_h = F_h.transpose(1, 2).unsqueeze(1).repeat(1,channels,1,1).contiguous() 
            F_h = F_h.view(batch_size*channels,F_h.shape[2],F_h.shape[3])
            F_w = F_w.unsqueeze(1).repeat(1,channels,1,1).contiguous()
            F_w = F_w.view(batch_size*channels,F_w.shape[2],F_w.shape[3])
            glimpses = glimpses.contiguous().view(batch_size*channels,image_h,image_w)
            glimpses = torch.bmm(F_h, glimpses)
            glimpses = torch.bmm(glimpses, F_w)
            glimpses = glimpses.view(batch_size, channels, self.glimpse_h, self.glimpse_w) # (B, c, glimpse_h, glimpse_w)

    
        else:
            glimpses = torch.bmm(F_h.transpose(1, 2), glimpses)
            glimpses = torch.bmm(glimpses, F_w)

        return glimpses  # (B, glimpse_h, glimpse_w)


class ARC(nn.Module):
    """
    This class implements the Attentive Recurrent Comparators. This module has two main parts.

    1.) controller: The RNN module that takes as input glimpses from a pair of images and emits a hidden state.

    2.) glimpser: A Linear layer that takes the hidden state emitted by the controller and generates the glimpse
                    parameters. These glimpse parameters are (h_center, w_center, delta). (h_center, w_center)
                    represents the relative position of the center of the glimpse on the image. delta determines
                    the zoom factor of the glimpse.

    Args:
        num_glimpses (int): How many glimpses must the ARC "see" before emitting the final hidden state.
        glimpse_h (int): The height of the glimpse in pixels.
        glimpse_w (int): The width of the glimpse in pixels.
        controller_out (int): The size of the hidden state emitted by the controller.

    """

    def __init__(self, num_glimpses: int=8, glimpse_h: int=8, glimpse_w: int=8, channels = 3,
                 controller_out: int=128) -> None:
        super().__init__()

        self.num_glimpses = num_glimpses
        self.glimpse_h = glimpse_h
        self.glimpse_w = glimpse_w
        self.controller_out = controller_out

        # main modules of ARC

        self.controller = nn.LSTMCell(input_size=(glimpse_h * glimpse_w * channels), hidden_size=self.controller_out)
        self.glimpser = nn.Linear(in_features=self.controller_out, out_features=3)

        # this will actually generate glimpses from images using the glimpse parameters.
        self.glimpse_window = GlimpseWindow(glimpse_h=self.glimpse_h, glimpse_w=self.glimpse_w)

    def forward(self, image_pairs: Variable) -> Variable:
        """
        The method calls the internal _forward() method which returns hidden states for all time steps. This i

        Args:
            image_pairs (B, 2, h, w):  A batch of pairs of images

        Returns:
            (B, controller_out): A batch of final hidden states after each pair of image has been shown for num_glimpses
            glimpses.

        """

        # return only the last hidden state
        all_hidden = self._forward(image_pairs)  # (2*num_glimpses, B, controller_out)
        last_hidden = all_hidden[-1, :, :]  # (B, controller_out)

        return last_hidden

    def _forward(self, image_pairs: Variable) -> Variable:
        """
        The main forward method of ARC. But it returns hidden state from all time steps (all glimpses) as opposed to
        just the last one. See the exposed forward() method.

        Args:
            image_pairs: (B, 2, h, w) A batch of pairs of images

        Returns:
            (2*num_glimpses, B, controller_out) Hidden states from ALL time steps.

        """

        # convert to images to float.
        image_pairs = image_pairs.float()

        # calculate the batch size
        batch_size = image_pairs.size()[0]

        # an array for collecting hidden states from each time step.
        all_hidden = []

        # initial hidden state of the LSTM.
        Hx = Variable(torch.zeros(batch_size, self.controller_out))  # (B, controller_out)
        Cx = Variable(torch.zeros(batch_size, self.controller_out))  # (B, controller_out)

        if use_cuda:
            Hx, Cx = Hx.cuda(), Cx.cuda()

        # take `num_glimpses` glimpses for both images, alternatingly.
        for turn in range(2*self.num_glimpses):
            # select image to show, alternate between the first and second image in the pair
            images_to_observe = image_pairs[:,  turn % 2]  # (B, c, h, w)

            # choose a portion from image to glimpse using attention
            glimpse_params = torch.tanh(self.glimpser(Hx))  # (B, 3)  a batch of glimpse params (x, y, delta)
            glimpses = self.glimpse_window.get_glimpse(images_to_observe, glimpse_params)  # (B, glimpse_h, glimpse_w)
            flattened_glimpses = glimpses.view(batch_size, -1)  # (B, glimpse_h * glimpse_w), one time-step

            # feed the glimpses and the previous hidden state to the LSTM.
            try:
                Hx, Cx = self.controller(flattened_glimpses, (Hx, Cx))  # (B, controller_out), (B, controller_out)
            except:
                a =1

            # append this hidden state to all states
            all_hidden.append(Hx)

        all_hidden = torch.stack(all_hidden)  # (2*num_glimpses, B, controller_out)

        # return a batch of all hidden states.
        return all_hidden


class ArcBinaryClassifier(nn.Module):
    """
    A binary classifier that uses ARC.
    Given a pair of images, feeds them the ARC and uses the final hidden state of ARC to
    classify the images as belonging to the same class or not.

    Args:
        num_glimpses (int): How many glimpses must the ARC "see" before emitting the final hidden state.
        glimpse_h (int): The height of the glimpse in pixels.
        glimpse_w (int): The width of the glimpse in pixels.
        controller_out (int): The size of the hidden state emitted by the controller.

    """

    def __init__(self, num_glimpses: int=8, glimpse_h: int=8, glimpse_w: int=8, channels = 3,
                 controller_out: int = 128):
        super().__init__()
        self.arc = ARC(
            num_glimpses=num_glimpses,
            glimpse_h=glimpse_h,
            glimpse_w=glimpse_w,
            channels = channels,
            controller_out=controller_out)

        # two dense layers, which take the hidden state from the controller of ARC and
        # classify the images as belonging to the same class or not.
        self.dense1 = nn.Linear(controller_out, 64)
        self.dense2 = nn.Linear(64, 1)

    def forward(self, image_pairs: Variable) -> Variable:
        arc_out = self.arc(image_pairs)

        d1 = F.elu(self.dense1(arc_out))
        decision = torch.sigmoid(self.dense2(d1))

        return decision

    def save_to_file(self, file_path: str) -> None:
        torch.save(self.state_dict(), file_path)


class CustomResNet50(nn.Module):

    def __init__(self, out_size=None):
        super(CustomResNet50, self).__init__()
        self.out_size = out_size
        self.resnet18 = models.resnet50(pretrained=True)
        if not(self.out_size is None):
            self.adaptativeAvgPooling = nn.AdaptiveAvgPool2d((out_size[0],out_size[1]))

    def forward(self, x):
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)
        x = self.resnet18.layer1(x)
        x = self.resnet18.layer2(x)
        x = self.resnet18.layer3(x) # 1024 x 14 x 14
        #x = self.resnet18.layer4(x) # 2048 x 7 x 7
        if not(self.out_size is None):
            x = self.adaptativeAvgPooling(x)
        return x
        


class CoAttn(nn.Module):
    def __init__(self, size = (19,19), num_filters = 1024, typeActivation = 'sum_abs', p = 2):
        """
        Initializes the naive ARC 
        """
        super(CoAttn, self).__init__()
        self.size = size
        self.size_pow = (pow(self.size[0],2),pow(self.size[1],2))
        self.typeActivation = typeActivation
        self.p = p
        self.num_filters = num_filters
        #self.W = nn.Parameter(torch.FloatTensor(self.size_pow[0],self.size_pow[1]).uniform_())
        self.W = nn.Parameter(torch.FloatTensor(self.num_filters,self.num_filters).uniform_())

    def forward(self, x):
        
        batch_size, npairs, nfilters, sizeFilterX, sizeFilterY = x.shape
        assert sizeFilterX == self.size[0]
        assert sizeFilterY == self.size[1]

        Q_a = x[:,0,:,:,:] # 10 x 2048 x 7 x 7
        Q_b = x[:,1,:,:,:] # 10 x 2048 x 7 x 7

        Q_a_1 = Q_a.view(batch_size, nfilters, self.size[0] * self.size[1]) # 20480 x 49
        Q_b_1 = Q_b.view(batch_size, nfilters, self.size[0] * self.size[1]) # 20480 x 49

        L = torch.bmm(torch.bmm(Q_b_1.transpose(2,1),self.W.unsqueeze(0).expand(batch_size,nfilters,nfilters)),Q_a_1)
        # normalize
        L_min = (L.min(dim=1)[0]).unsqueeze(2).expand(batch_size,sizeFilterX*sizeFilterY,sizeFilterX*sizeFilterY)
        L_max = (L.max(dim=1)[0]).unsqueeze(2).expand(batch_size,sizeFilterX*sizeFilterY,sizeFilterX*sizeFilterY)
        L_norm = (L - L_min) / (L_max - L_min)
        # produce attention weights
        A_a = torch.nn.Softmax(dim=2)(L_norm)
        A_b = torch.nn.Softmax(dim=2)(L_norm.transpose(2,1))
        # attention summaries
        Z_a = torch.bmm(Q_a_1,A_a)
        Z_b = torch.bmm(Q_b_1,A_b)
        # resize the results
        Z_a = Z_a.view(batch_size,nfilters,sizeFilterX,sizeFilterY)
        Z_b = Z_b.view(batch_size,nfilters,sizeFilterX,sizeFilterY)


        x = torch.cat((Z_a.unsqueeze(1),Z_b.unsqueeze(1)),1)
        return x

        
        
        
class FullContextARC(nn.Module):
    def __init__(self, hidden_size, num_layers, vector_dim):
        """
        Initializes a multi layer bidirectional LSTM
        :param hidden_size: the neurons per layer 
        :param num_layers: number of layers                            
        :param batch_size: The experiments batch size
        """
        super(FullContextARC, self).__init__()
        self.lstm = nn.LSTM(input_size=vector_dim,
                            num_layers=num_layers,
                            hidden_size=hidden_size,
                            bidirectional=True,
                            batch_first = True)
        #self.dense1 = nn.Linear(hidden_size*2, 64)
        #self.dense2 = nn.Linear(64, 1)
        self.dense1 = nn.Linear(hidden_size * 2, 1)
        #self.relu = nn.ReLU()
        #self.logSoftmax = nn.LogSoftmax()

    def forward(self, x):
        ## bidirectional lstm
        x, (hn, cn) = self.lstm(x)
        #x = F.elu(self.dense1(x)).squeeze()
        x = torch.sigmoid(self.dense1(x).squeeze())
        #x = self.dense2(x).squeeze()
        #x = self.relu(x)
        #x = self.logSoftmax(x)
        return x