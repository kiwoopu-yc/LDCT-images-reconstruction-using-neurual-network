# LDCT-images-reconstruction-using-neurual-network
Practice module for isy5004

## What is LDCT?
During the medical CT scan, the radiating equipment radiates X-rays through the human body at first. During this process, X-rays will attenuate to different degrees due to different tissue structures of the human body, and then the attenuated X-rays will be received by the instrument again. The information received by the instrument is the result of Radon transform.So to get the image that the doctor is doing, we need to reconstruct the profile of the human body that the X-ray has passed through from the results of the Radon transform, and this reconstruction process is called the inverse Radon transform.The projection information obtained by physical sensors and detectors may have noise, so it is not easy for inverse Radon transform or image reconstruction.The conventional reconstruction process is consistent with the Radon transform through many intermediate layers, but each stage highly depends on the results of the previous stage, and each step in the reconstruction process will have a great impact on the reconstruction image, so the algorithm optimization of each stage is also the current research hotspot.Specifically, we build a special neural network to simulate the process of filtering backprojection reconstruction, and its architecture can be divided into two parts.In the first part, the learnable fully connected filter layer is used to filter the Radon projection in the perspective direction, and then the learnable sinusoidal back projection layer is used to transform the filtered Radon projection into an image.The second part is a common neural network structure, which is used to further improve the reconstruction performance of image domain.

## Mathematical theory
### Fourier Transform
The mathematical meaning of the Fourier transform is to convert a function into a set of periodic functions.All periodic functions can be expressed as the product of a sum of sines (or cosines) and a Fourier series.The physical meaning of Fourier transform is to realize the domain transformation of images. Fourier transform and inverse Fourier transform respectively transform (gray scale) images from spatial domain to frequency domain and then from frequency domain to spatial domain.
The Fourier transform in a two-dimensional plane:
<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;F(\xi,\eta)" title="F(\xi,\eta)" /> is the result of f(x,y) after two-dimensional Fourier transform.

<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;F\left\{f(x,y)\right\}&space;&space;=&space;F(\xi,\eta)&space;=&space;\int\int_{-\infty}^{\infty}f(x,y)e^{-j2\pi(\xi&space;x&plus;\eta&space;y)}dxdy" title="F\left\{f(x,y)\right\} = F(\xi,\eta) = \int\int_{-\infty}^{\infty}f(x,y)e^{-j2\pi(\xi x+\eta y)}dxdy" />

<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;F^{-1}\left\{F(\xi,\eta)\right\}&space;&space;=&space;f(x,y)&space;=&space;\int\int_{-\infty}^{\infty}F(\xi,\eta)e^{j2\pi(\xi&space;x&plus;\eta&space;y)}d\xi&space;d\eta" title="F^{-1}\left\{F(\xi,\eta)\right\} = f(x,y) = \int\int_{-\infty}^{\infty}F(\xi,\eta)e^{j2\pi(\xi x+\eta y)}d\xi d\eta" />


## Step1: Get the radon projection from Astra Toolbox using matlab
### About Astra Toolbox
Astra Toolbox is a tool kit for 2D and 3D tomography using GPU acceleration. It is available for download on either MATLAB or Python. The authors of Astra Toolbox have a column on YouTube that provides tutorial videos for beginners.
Astra Toolbox supports 2D parallel and fan beam projections as well as 3D parallel and cone beam projections, and provides a range of reconstruction algorithms, including: filtered back projection reconstruction (FBP), joint iterative reconstruction (SIRT), joint algebraic iterative reconstruction (SART), and conjugate gradient method (CGLS).The authors also provide a variety of different filters for each algorithm.

:rocket:[Astra Toolbox](https://www.youtube.com/channel/UCcnUlWuFeS9miMR_o8-6v3w/videos)
### Using Altra Toolbox Obtain Radon Projections

#### Data prepration: Resize
All the training test data were processed in advance in MATLAB. The images in the data set were first clipped to a uniform size (512×512), and then processed using Astra Toolbox. For convenience I used matlab.

:rocket:[Check batch_resize_pic.m in this folder](https://github.com/kiwoopu-yc/LDCT-images-reconstruction-using-neurual-network/tree/main/Astra)

#### Data prepration: Obtain pojections with different views
Lower views is meaning lower dose in CT.
Radon projections of images were generated using Astra Toolbox with the number of perspectives 72, 145, 290, and 1160, the number of detectors 736, and the data was stored in .mat format.

:rocket:[Check batch_create_view.m in this folder](https://github.com/kiwoopu-yc/LDCT-images-reconstruction-using-neurual-network/tree/main/Astra)

:rocket:[Check batch_create_sinograme.m in this folder](https://github.com/kiwoopu-yc/LDCT-images-reconstruction-using-neurual-network/tree/main/Astra)

#### Data prepration: Obtain images after FBP algorithm for comparison
The filter back projection reconstruction (FBP) image of the image was generated by Astra Toolbox for reference. The size of both the input image and the reconstructed image were 512×512, and the data storage format was .mat.

:rocket:[Check batch_create_FBP.m in this folder](https://github.com/kiwoopu-yc/LDCT-images-reconstruction-using-neurual-network/tree/main/Astra)

In Python, the function in the mat4py package is used to read. Mat file, and the Radon projection and reference image of the same image are put into a tuple as a training set, and stored as a file in the format of .PKL.
In Python, functions in the mat4py package are used to read the .mat file, and the Radon projection of the image is stored as the .npy format file as the test set.

## Step2: Train the neural network

### Architecture

<div style="align: center">
<img src="https://github.com/kiwoopu-yc/LDCT-images-reconstruction-using-neurual-network/blob/main/Pics/archi.jpg"/>
</div>

### Neural network using PyTorch

#### VAE

Variational Autoencoder (VAE) is an extension of AE. It will have a sampling layer instead of a simple full connection layer. This layer will use the mean and variance from the last layer of the encoder to get a Gaussian sample and use it as the input to the decoder. 

:rocket: [Define VAE (line 49-81)](https://github.com/kiwoopu-yc/LDCT-images-reconstruction-using-neurual-network/blob/main/Model/VAE.py)

#### 2D-CNN(unfinished)

:rocket: [2D-CNN](https://github.com/kiwoopu-yc/LDCT-images-reconstruction-using-neurual-network/blob/main/Model/2dcnn(unfinished).py)

#### Dataset
The training dataset consists of a large number of generic images from ImageNet.

The test dataset consisted of 20 real clinical brain CT images from Kaggle's public dataset, and the remaining 180 brain CT images from the same dataset were also included in the training set. Your can find in the root file. 

#### Load data

:rocket: step1: [Convert .mat to numpy](https://github.com/kiwoopu-yc/LDCT-images-reconstruction-using-neurual-network/blob/main/Model/mat2np.py)

:rocket: step2: [Compile training data and target data](https://github.com/kiwoopu-yc/LDCT-images-reconstruction-using-neurual-network/blob/main/Model/2npyconnect.py)

:rocket: step3: [Load data using torch.utils.data.DataLoader (line 117-119)](https://github.com/kiwoopu-yc/LDCT-images-reconstruction-using-neurual-network/blob/main/Model/VAE.py)

#### Loss function and Optimizer


:rocket: [Training (line 122-149)](https://github.com/kiwoopu-yc/LDCT-images-reconstruction-using-neurual-network/blob/main/Model/VAE.py)


Adjust parameters based on results.

Here is the loss using different parameters, we can see Adam worked best at end.
<div style="align: center">
<img src="https://github.com/kiwoopu-yc/LDCT-images-reconstruction-using-neurual-network/blob/main/Pics/optimizers.jpg"/>
</div>

#### Evaluation and visulization



<div style="align: center">
<img src="https://github.com/kiwoopu-yc/LDCT-images-reconstruction-using-neurual-network/blob/main/Pics/72view.jpg"/>
</div>



<div style="align: center">
<img src="https://github.com/kiwoopu-yc/LDCT-images-reconstruction-using-neurual-network/blob/main/Pics/all.jpg"/>
</div>


