#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110

#include <CL/cl.h>
#include "cl2.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>

const size_t BLOCK_SIZE = 256;


void prop_hillis_steele(uint array_size, double* input, double *chunks, cl::Context& context, cl::CommandQueue& queue, cl::Program& program) {
    cl::Kernel kernel_prop(program, "prop_hillis_steele");
    cl::Buffer dev_input (context, CL_MEM_READ_WRITE, sizeof(double) * array_size);
    cl::Buffer dev_chunks (context, CL_MEM_READ_ONLY, sizeof(double) * array_size / BLOCK_SIZE);
    // copy from cpu to gpu
    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(double) * array_size, &input[0]);
    queue.enqueueWriteBuffer(dev_chunks, CL_TRUE, 0, sizeof(double) * array_size / BLOCK_SIZE, &chunks[0]);

    kernel_prop.setArg(0, dev_input);
    kernel_prop.setArg(1, dev_chunks);

    queue.enqueueNDRangeKernel(kernel_prop, cl::NullRange, cl::NDRange(array_size), cl::NDRange(BLOCK_SIZE));
    queue.enqueueReadBuffer(dev_input, CL_TRUE, 0, sizeof(double) * array_size, &input[0]);
}

void scan_hillis_steele(uint array_size, double* input, double *output, cl::Context& context, cl::CommandQueue& queue, cl::Program& program) {

    // allocate device buffer to hold message
    cl::Buffer dev_input (context, CL_MEM_READ_ONLY, sizeof(double) * array_size);
    cl::Buffer dev_output(context, CL_MEM_READ_WRITE, sizeof(double) * array_size);

    // copy from cpu to gpu
    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(double ) * array_size, &input[0]);

    // load named kernel from opencl source
    cl::Kernel kernel_scan(program, "scan_hillis_steele");
    kernel_scan.setArg(0, dev_input);
    kernel_scan.setArg(1, dev_output);
    kernel_scan.setArg(2, cl::Local(sizeof(double) * BLOCK_SIZE));
    kernel_scan.setArg(3, cl::Local(sizeof(double) * BLOCK_SIZE));
    queue.enqueueNDRangeKernel(kernel_scan, cl::NullRange, cl::NDRange(array_size), cl::NDRange(BLOCK_SIZE));

    // get output
    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(double) * array_size, &output[0]);

    // recursive call

    if (array_size > BLOCK_SIZE) {
        uint new_array_size = ceil((double)array_size / BLOCK_SIZE);
        uint round_new_array_size = ceil((double)new_array_size / BLOCK_SIZE) * BLOCK_SIZE;
        double new_input[round_new_array_size];
        for (int i = 0; i < new_array_size; i++) {
            new_input[i] = output[(i + 1) * BLOCK_SIZE - 1];
        }
        double new_output[round_new_array_size];
        scan_hillis_steele(round_new_array_size, new_input, new_output, context, queue, program);
        prop_hillis_steele(array_size, output, new_output, context, queue, program);
    }
}


int main()
{
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {
        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("scan.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, cl_string);

        // create program
        cl::Program program(context, source);

        // compile opencl source
        try
        {
            program.build(devices);
        }
        catch (cl::Error const & e)
        {
            std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cout << log_str;
            return 0;
        }

        uint array_size;
        std::ifstream infile("../input.txt", std::ios_base::in);
        infile >> array_size;

        uint round_array_size = ceil((double)array_size / BLOCK_SIZE) * BLOCK_SIZE;

        double input[round_array_size];
        for (size_t i = 0; i < array_size; ++i) {
            infile >> input[i];
        }

        double output[round_array_size];
        scan_hillis_steele(round_array_size, input, output, context, queue, program);
        std::ofstream outfile("../output.txt");
        for (size_t i = 0; i < array_size; ++i)
        {
            outfile << output[i] << " ";
        }
    }
    catch (cl::Error const & e)
    {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }

    return 0;
}
