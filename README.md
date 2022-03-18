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

## Exercise2

Now we check the timings if we make the size 10 times bigger and we change the device from the CPU to GPU:

CPU

![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/runoncpu10times.png "")

GPU

![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/gpumatrixmulttiming.png "")

Someone would wonder here why we get similar results. The reason is that the matrix multiplication of the INTEL sample is as naive as it gets.

The main reason this doesn't perform so well is because we are accessing the GPU's off-chip memory way too much. We can count:

to do the (N * N * N) multiplications and additions, we need 2(N * N * N) loads and N*N stores. Since the multiplications and additions can actually be fused into a single hardware instruction (FMA), the computational intensity of the code is only 0.5 instructions per memory access. This is bad.

## Exercise3
The code is in the /src directory.

To run the implementation:

clean the build folder:

`cd src/build`

`rm -rf *`

Then locate your dpcpp compiler:

`dpcpp --version`

Then run cmake with the appropriate path:

`cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/latest/linux/bin/dpcpp -DCMAKE_CXX_FLAGS="-fsycl"`

Then make:

`make -B`

And then you can run it (default size 1024, but you can change it by passing another size):

`./dpc_exercise 1344`

Here are some examples that show same results in CPU and GPU:
![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/size1024.png "")

![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/size1708.png "")

Here we should note that for this implementation and the used GPU, the resolution is until a size of N = 1708. If we grow the size bigger we can see that we have an overflow and get negative values (But the same on both sides).


## Exercise4
After checking some of the VTune results I saw that I was initiating wrong grids of work-items. For example below we can see that the 2 last kernels where not generating the optimal number of work-items at their peak:

![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/wrongNDrange.png "")

 After modifying the src code to use the ndrange call (similar to OpenCL) which I am more familiar we can see that at least the 3 kernels are set up correctly:

![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/correctNDrange.png "")

### Possible Improvements
### Memory Diagrams

First we check the Memory Hierarchy Diagrams of each kernel for a first idea:

1st Kernel:

![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/Memdiagram1.png "")

2nd Kernel:

![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/Memdiagram2.png "")

3rd Kernel:

![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/Memdiagram3.png "")

For the first 2 kernels I can see that even though vtune does not complain about it, that we use a lot of the L3 cache. We could try to fit the vectors in the SLM which right now is not used at all.

Something really strange is that for the last kernel which is the reduction, vtune says that it did nothing. How can a reduction be so quick? :/

### GPU HW Threads dispatch

Another observation I could see is that the first kernel does not uses all of the available hardware threads available. At the peak, only uses 64% of them. That can be seen below:

![alt text](https://github.com/nikosandriotis/gsoc-2022-warm-up/blob/main/snapshots/Threaddispatch.png "")

This can be solved with better NDrange setup. Using the local_size instead of only global_size would improve this by using tiling in the multiplication kernels.

## IMPORTANT NOTES

It seems that the reduction I implemented, after checking VTune is not working correctly but I am still getting correct results at the end! That means either Vtune shows funny results or I am missing something.

It seems that I have an issue with my setup and I could not run it. Unfortunately I cannot update my versions of the compilers and drivers because this is my work's laptop. All of VTune happened in another laptop that I could find and updated everything related to oneapi and intel.
