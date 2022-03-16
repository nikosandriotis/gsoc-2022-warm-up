//==============================================================
// Vector Add is the equivalent of a Hello, World! sample for data parallel
// programs. Building and running the sample verifies that your development
// environment is setup correctly and demonstrates the use of the core features
// of DPC++. This sample runs on both CPU and GPU (or FPGA). When run, it
// computes on both the CPU and offload device, then compares results. If the
// code executes on both CPU and offload device, the device name and a success
// message are displayed. And, your development environment is setup correctly!
//
// For comprehensive instructions regarding DPC++ Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide and search based on
// relevant terms noted in the comments.
//
// DPC++ material used in the code sample:
// •	A one dimensional array of data shared between CPU and offload device.
// •	A device queue and kernel.
//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <dpc_common.hpp>
//#include <cl_utils.h>
#if FPGA || FPGA_EMULATOR
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

using namespace sycl;

// Array size for this example.
size_t array_size = 4;

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

//************************************
// Vector add in DPC++ on device: returns sum in 4th parameter "sum".
//************************************
void vecMatvec(queue& q,
    const int* v_host,
    const int* b_host,
    int* c_gpu,
    int* res_gpu,
    int* sum,
    size_t N) {
    /*
        To Multiply: C[M][P] = A[M][N] * B[N][P]
    */
    try {

        //buffer<int, 1> v(v_host.data(), range<1>{v_host.size()}); // 1D vector v
        //buffer<int, 2> b(b_host.data(), range<2>{N, N}); // 2D buffer for matrix b
        //buffer<int, 1> c(c_gpu.data(), range<1>{c_gpu.size()}); // 1D vector intermediate
        //buffer<int, 1> res(res_gpu.data(), range<1>{res_gpu.size()}); // 1D vector for partial results

        std::cout <<("Starting computing on GPU...\n");
        std::cout << "GPU::Multiplying V^T and B into C.\n";
        auto firstkernel = q.submit([&](handler& h) {

            //auto V = v_host.get_access<access::mode::read>(h);
            //auto B = b_host.get_access<access::mode::read>(h);
            //auto C = c_gpu.get_access<access::mode::write>(h);
           std::cout << "GPU::Hola.\n";
            h.parallel_for(range<2>{N, N}, [=](id<2> index) {
                // Threading index that iterates over c.
                int row = index[0];
                int col = index[1];
                //auto sum = 0;
                // Compute result of ONE element of C
                //for (int i = 0; i < N; i++)
                c_gpu[row * N + col] = v_host[col] * b_host[row * N + col];
                });
            });

        std::cout << "GPU::Multiplying C and V into RES.\n";
        auto secondkernel = q.submit([&](handler& h) {

            //auto C = c_gpu.get_access<access::mode::read>(h);
            //auto V = v_host.get_access<access::mode::read>(h);
            //auto RES = res_gpu.get_access<access::mode::write>(h);

            h.parallel_for(N, [=](id<1> index) {
                // Threading index that iterates over c.
                int id = index[0];
                // Compute result of ONE element of C
                for (int col = 0; col<N; col++)
                res_gpu[id] += v_host[id] * c_gpu[id * N + col];
                });
            });

        std::cout << "GPU::Reduction happening.\n";
        auto reduction = q.submit([&](handler& h) {
            //auto RES = res_gpu.get_access<access::mode::read>(h);

            h.parallel_for(N, [=](id<1> i) {
              sycl::ONEAPI::atomic_ref<int,
                                       sycl::ONEAPI::memory_order::relaxed,
                                       sycl::ONEAPI::memory_scope::system,
                                       access::address_space::global_space>
                                       (*sum) += res_gpu[i]
                ;});
            });
        reduction.wait();
    }
    catch (sycl::exception const& e) {
        std::cout << "An exception is caught while multiplying in GPU.\n";
        std::terminate();
    }
}

void initData(int* v, int* b, size_t size){
  // initialize vector
  for (int i=0; i<size; i++) v[i] = i;
  // initialize Matrix
  for (int i=0; i<size; i++){
    for (int j=0; j<size; j++){
      b[i*size+j] = i+j;
    }
  }
}

int CPU_calculation(int* v, int* b, size_t size) {
  //we have to compute (v^T * b * v)
  int expected_sum = 0.0;
  for (int i=0; i < size; i++){
      for (int j=0; j < size; j++){
          expected_sum += v[i] * b[size*i+j] * v[j];
        }
  }
  return expected_sum;
}
//************************************
// Demonstrate vector add both in sequential on CPU and in parallel on device.
//************************************
int main(int argc, char* argv[]) {
  // Change array_size if it was passed as argument
  if (argc > 1) array_size = std::stoi(argv[1]);
  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  INTEL::fpga_emulator_selector d_selector;
#elif FPGA
  // DPC++ extension: FPGA selector on systems with FPGA card.
  INTEL::fpga_selector d_selector;
#else
  // The default device selector will select the most performant device.
  default_selector d_selector;
#endif

  try {
    constexpr size_t N = 4;

    queue q(d_selector, exception_handler);
    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "N is: " << N << "\n";

    // Create arrays with "array_size" to store input and output data. Allocate
    // unified shared memory so that both CPU and device can access them.
    //std::vector<int> v(array_size);
    //std::vector<int> b(array_size * array_size);
    //std::vector<int> c_gpu(array_size * array_size);
    //std::vector<int> res_gpu(array_size);

    int *v = malloc_shared<int>(N, q);
    int *b = malloc_shared<int>(N*N, q);
    int *c_gpu = malloc_shared<int>(N*N, q);
    int *res_gpu = malloc_shared<int>(N, q);
    int* sum = malloc_shared<int>(1, q); //final result
    *sum = 0;

    initData(v, b, N);
    int expected_sum = CPU_calculation(v, b, N);
    // Vector addition in DPC++.
    vecMatvec(q, v, b, c_gpu, res_gpu, sum, N);
    std::cout << "GPU_RESULT:"<< *sum << "\n";
    std::cout << "EXPECTED_RESULT:"<<expected_sum << "\n";
    // Verify that the two arrays are equal.
      if (*sum != expected_sum) {
        std::cout << "Vector add failed on device.\n";
        return -1;
      }

    free(sum, q);
    free(v, q);
    free(b, q);
    free(c_gpu, q);
    free(res_gpu, q);

  } catch (exception const &e) {
    std::cout << "An exception is caught while adding two vectors.\n";
    std::terminate();
  }

  std::cout << "Vector add successfully completed on device.\n";
  return 0;
}
