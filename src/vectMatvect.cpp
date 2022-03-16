#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <dpc_common.hpp>
//#include <cl_utils.h>
#if FPGA || FPGA_EMULATOR
#include <CL/sycl/INTEL/fpga_extensions.hpp>
#endif

using namespace sycl;

size_t N = 1024;
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
// (v^T * b * v) DPC++ on device:
//************************************
void vecMatvec(queue& q,
    const int* v_host,
    const int* b_host,
    int* c_gpu,
    int* res_gpu,
    int* sum,
    size_t N) {
    try {
        std::cout <<("Starting computing on GPU...\n");
        std::cout << "GPU::Multiplying V^T and B into C.\n";
        auto firstkernel = q.submit([&](handler& h) {
           std::cout << "GPU::Hola.\n";
            h.parallel_for(range<2>{N, N}, [=](id<2> index) {
                // Threading index that iterates over c_gpu.
                int row = index[0];
                int col = index[1];
                //important here to have the unique indexing of c_gpu
                c_gpu[row * N + col] = v_host[col] * b_host[row * N + col];
                });
            });

        std::cout << "GPU::Multiplying C and V into RES.\n";
        auto secondkernel = q.submit([&](handler& h) {
            h.parallel_for(N, [=](id<1> index) {
                // Threading index that iterates over res_gpu.
                int id = index[0];
                //with the for loop here we avoid a necessary reduction here
                for (int col = 0; col<N; col++)
                res_gpu[id] += v_host[id] * c_gpu[id * N + col];
                });
            });
        std::cout << "GPU::Reduction happening.\n";
        //sycl reduction implementation
        auto reduction = q.submit([&](handler& h) {
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

//initialize the objects
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

//verification on CPU
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

int main(int argc, char* argv[]) {
  // Change array_size if it was passed as argument
  if (argc > 1) N = std::stoi(argv[1]);
  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  INTEL::fpga_emulator_selector d_selector;
#elif FPGA
  // DPC++ extension: FPGA selector on systems with FPGA card.
  INTEL::fpga_selector d_selector;
#else
  // The default device selector will select the most performant device.
  gpu_selector d_selector;
#endif

  try {
    queue q(d_selector, exception_handler);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<info::device::name>() << "\n";
    std::cout << "N is: " << N << "\n";

    //shared variables so we can access them from cpu and gpu
    int *v = malloc_shared<int>(N, q);
    int *b = malloc_shared<int>(N*N, q);
    int *c_gpu = malloc_shared<int>(N*N, q);
    int *res_gpu = malloc_shared<int>(N, q);
    int* sum = malloc_shared<int>(1, q); //final result
    *sum = 0;

    initData(v, b, N);
    int expected_sum = CPU_calculation(v, b, N);

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
