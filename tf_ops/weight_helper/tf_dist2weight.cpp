#include <cmath> // sqrtf
#include <cuda.h>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

REGISTER_OP("BuildSampleWeight")
    .Attr("radius: float")          // range search radius: required in bilinear_like weight computation
    .Attr("weight_type: int")      // it determines the computation strategy of the weight
    .Input("nn_count: int32")      // number of neighbors: batch * mpoint
    .Input("nn_dist: float32")     // distance to the neighbors: batch * mpoint * nn_sample
    .Output("weight: float32")     // interpolation weights of the neighbors: batch * mpoint * nn_sample
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(1));
        return Status::OK();
    });


void buildSampleWeightLauncher(int B, int M, int nnSample, int weightType, float radius,
                               const int* nnCount, const float* nnDist, float* Weight);
class BuildSampleWeightGpuOp : public OpKernel {
    public:
        explicit BuildSampleWeightGpuOp(OpKernelConstruction* context) : OpKernel(context)
        {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0, errors::InvalidArgument("Range search requires radius>0, got ", radius_));

            OP_REQUIRES_OK(context, context->GetAttr("weight_type", &weight_type_));
            OP_REQUIRES(context, weight_type_!=0 || weight_type_!=1,
                        errors::InvalidArgument("weight_type to be chosen among {0,1} ", weight_type_));
        }
        void Compute(OpKernelContext* context) override {
            // Grab the input tensors
            const Tensor& nn_count_tensor = context->input(0);
            const Tensor& nn_dist_tensor = context->input(1);

            // get the dims required by computations
            int B = nn_dist_tensor.shape().dim_size(0); // batch size
            int M = nn_dist_tensor.shape().dim_size(1); // number of database points
            int nn_sample = nn_dist_tensor.shape().dim_size(2);  // number of neighbors of the query points

            OP_REQUIRES(context, nn_count_tensor.dims()==2, errors::InvalidArgument("Shape of nnCount requires to be (batch, mpoint)"));
            OP_REQUIRES(context, nn_dist_tensor.dims()==3, errors::InvalidArgument("Shape of nn_dist should be (batch, mpoint, nn_sample)"));

            // flatten the input tensors
            auto nn_count_flat = nn_count_tensor.flat<int32>();
            auto nn_dist_flat = nn_dist_tensor.flat<float>();

            // Create an output tensor
            Tensor* weight_tensor = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{B,M,nn_sample}, &weight_tensor));
            auto weight_flat = weight_tensor->flat<float>();

            const int* nnCount = &(nn_count_flat(0));
            const float* nnDist = &(nn_dist_flat(0));
            float* Weight = &(weight_flat(0));

            cudaMemset(Weight, 0, sizeof(float)*B*M*nn_sample);

            buildSampleWeightLauncher(B, M, nn_sample, weight_type_, radius_, nnCount, nnDist, Weight);
        }
    private:
        float radius_;
        int weight_type_;
};
REGISTER_KERNEL_BUILDER(Name("BuildSampleWeight").Device(DEVICE_GPU), BuildSampleWeightGpuOp);







