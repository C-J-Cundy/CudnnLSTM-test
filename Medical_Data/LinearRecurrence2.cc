#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"


using namespace tensorflow;

REGISTER_OP("LinearRecurrenceNew")
    .Input("f: float32")
    .Input("g: float32")
    .Output("h: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
      });

class LinearRecurrenceOp : public OpKernel {
 public:
  explicit LinearRecurrenceOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor1 = context->input(0);
    auto f = input_tensor1.flat<float>();

    const Tensor& input_tensor2 = context->input(1);
    auto b = input_tensor2.flat<float>();    

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor1.shape(),
                                                     &output_tensor));
    auto output = output_tensor->flat<float>();

    // Set all but the first element of the output tensor to 0.
    const int N = f.size();
    const int stride = input_tensor1.shape().dim_size(0);      
    const int stride1 = input_tensor1.shape().dim_size(2);
    for (int i = 0; i < stride1; i++) {
      output(i) = b(i);
    }
    for (int i = 1; i < N / stride1; i++) {
      for (int j = 0; j < stride1; j++) {
	output(i*stride1 + j) = output((i-1)*stride1 + j) * f(i*stride1 + j) + b(i*stride1 + j);
	//	output(i*stride1 + j) = 0;
      }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("LinearRecurrenceNew").Device(DEVICE_CPU), LinearRecurrenceOp);
