# LDCT-images-reconstruction-using-neurual-network
Practice module for isy5004

## What is LDCT?
During the medical CT scan, the radiating equipment radiates X-rays through the human body at first. During this process, X-rays will attenuate to different degrees due to different tissue structures of the human body, and then the attenuated X-rays will be received by the instrument again. The information received by the instrument is the result of Radon transform.So to get the image that the doctor is doing, we need to reconstruct the profile of the human body that the X-ray has passed through from the results of the Radon transform, and this reconstruction process is called the inverse Radon transform.The projection information obtained by physical sensors and detectors may have noise, so it is not easy for inverse Radon transform or image reconstruction.The conventional reconstruction process is consistent with the Radon transform through many intermediate layers, but each stage highly depends on the results of the previous stage, and each step in the reconstruction process will have a great impact on the reconstruction image, so the algorithm optimization of each stage is also the current research hotspot.Specifically, we build a special neural network to simulate the process of filtering backprojection reconstruction, and its architecture can be divided into two parts.In the first part, the learnable fully connected filter layer is used to filter the Radon projection in the perspective direction, and then the learnable sinusoidal back projection layer is used to transform the filtered Radon projection into an image.The second part is a common neural network structure, which is used to further improve the reconstruction performance of image domain.

## Mathematical theory
### Fourier Transform
The mathematical meaning of the Fourier transform is to convert a function into a set of periodic functions.All periodic functions can be expressed as the product of a sum of sines (or cosines) and a Fourier series.The physical meaning of Fourier transform is to realize the domain transformation of images. Fourier transform and inverse Fourier transform respectively transform (gray scale) images from spatial domain to frequency domain and then from frequency domain to spatial domain.The Fourier transform in a two-dimensional plane:

$$
F\left\{f(x,y)\right\}  
= F(\xi,\eta) 
= \int\int_{-\infty}^{\infty}f(x,y)e^{-j2\pi(\xi x+\eta y)}dxdy
$$



## Step1: Get the radon projection from AstraToolbox using matlab


