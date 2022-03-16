# gsoc-2022-warm-up

This is the repository for the warm-up exercise of Google Summer of Code 2022, for the ATHENA project titled: "HSF EIC ATHENA Clustering on GPUs"
## Exercise1

First we install the prerequisites (compilers, gpu drivers etc) for running DPCPP on our machine.
Then we can install and launch oneapi-cli tool. After that we can generate the 2 samples for vector addition and matrix multiplication.

Generating the samples:
![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/oneapicli1st.png "")

![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/oneapicli2ndscreenshot.png "")

Building the vector sample:
![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/buildingvectoradd1.png "")

![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/buildingvector2.png "")

And the unified shared memory version:
![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/buildingvector3.png "")

![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/buildingvector4.png "")


Building the matrix sample. Here also I can see that my compiler version is outdated too:
![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/buildingmatrixmult1.png "")

![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/buildingmatrixmult2.png "")

##Exercise2

Now we check the timings if we make the size 10 times bigger and we change the device from the CPU to GPU:
![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/matrixmultwithCPUandx10.png"")

![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/matrixmultmultitime2nd_normaltimes10.png "")

##Exercise3
The code is in the /src directory.

To run the implementation:

clean the build folder:
`cd src/build`
`rm -rf *`

Then locate your dpcpp compiler:

`dpcpp --version`

Then run cmake with the appropriate path:

`cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2021.3.0/linux/bin/dpcpp -DCMAKE_CXX_FLAGS="-fsycl"`

Then make:

`make -B`

And then you can run it (default size 1024, but you can change it in line 119 of src/vectMatvect.cpp):

`./dpc_exercise`

Here are some examples that show same results in CPU and GPU:
![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/matrixmultwithCPUandx10.png"")

![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/matrixmultmultitime2nd_normaltimes10.png "")

##IMPORTANT NOTES

Even though I was expecting VTune to be the easiest part of the assignment (I have used it extensively), it seems that I have an issue with my setup and I could not run it. Unfortunately I cannot update my versions of the compilers and drivers because this is my work's laptop.

I tried on another laptop but it was not compatible since it has a Gen8 gpu (min Gen9 required for running oneapi on GPU). Here is the issue when trying to run it on GPU (I was able in CPU):

![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/erroronothermachine.jpg "")

However what is needed in VTune is to run a gpu-offload like so:
`vtune -collect gpu-offload -allow-multiple-runs -analyze-system -finalization-mode=full ./dpc_exercise`

And then most probably with the "naive" implementation that we have here, we should check the platform schematic to see how often we missed the L3 and went to the main memory. An improvement for this would be to do Tiling and use also local identifiers in the GPU and not only the global ones. We can get much more performance by manually caching sub-blocks of the matrices (tiles) in the GPU's on-chip local memory (== shared memory in CUDA).
