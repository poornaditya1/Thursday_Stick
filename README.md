# Thursday_Stick

Visually challenged people face a lot of difficulty during their  locomotion , they find it difficult to figure out the objects/obstacles in front of them. So we have designed an escorting Stick.
Our solution is a partial hardware hack, it uses ResNet model to detect the object in front of the camera.
It can classify 1000 different objects and it reads out the name of the object. 


## Demo Prototype

![WhatsApp Image 2020-09-29 at 22 15 43](https://user-images.githubusercontent.com/62421629/94591671-fd738d80-02a5-11eb-929c-27d25d1d11ac.jpeg)

## Working of the model 

The camera on the walking stick captures the image of the obstacle ahead. Then our code, classifies the image and reads out the name of the obstacle to the user so he/she can act accordingly

# ResNet50

A great visualisation of the 50 layers of the ResNet model can be found [here](https://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006)

### Architecture of ResNet50 [1]

The architecture of ResNet50 has 4 stages as shown in the diagram below. The network can take the input image having height, width as multiples of 32 and 3 as channel width. For the sake of explanation, we will consider the input size as 224 x 224 x 3. Every ResNet architecture performs the initial convolution and max-pooling using 7×7 and 3×3 kernel sizes respectively. Afterward, Stage 1 of the network starts and it has 3 Residual blocks containing 3 layers each. The size of kernels used to perform the convolution operation in all 3 layers of the block of stage 1 are 64, 64 and 128 respectively. The curved arrows refer to the identity connection. The dashed connected arrow represents that the convolution operation in the Residual Block is performed with stride 2, hence, the size of input will be reduced to half in terms of height and width but the channel width will be doubled. As we progress from one stage to another, the channel width is doubled and the size of the input is reduced to half.

For deeper networks like ResNet50, ResNet152, etc, bottleneck design is used. For each residual function F, 3 layers are stacked one over the other. The three layers are 1×1, 3×3, 1×1 convolutions. The 1×1 convolution layers are responsible for reducing and then restoring the dimensions. The 3×3 layer is left as a bottleneck with smaller input/output dimensions.

Finally, the network has an Average Pooling layer followed by a fully connected layer having 1000 neurons (ImageNet class output).

![Architecture of ResNet50](https://cv-tricks.com/wp-content/uploads/2019/07/ResNet50_architecture-1.png)

# Story behind the name:

Just like Iron Man had his own AI assistant "FRIDAY" (after "JARVIS"), our solution is a smart assistant for visually challenged people that can assist them in their daily lives. Obviously, our assistant isn't as advanced as "FRIDAY", so we present "Thursday_Stick", an assistant at a level lower than the super-smart "FRIDAY" 


# References

1. https://cv-tricks.com/keras/understand-implement-resnets/
