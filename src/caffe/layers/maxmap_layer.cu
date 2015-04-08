#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxMapForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int cur_num = index / channels;
    int cur_channel = index % channels;
    const Dtype* cur_bottom = bottom_data +
        (cur_num * channels + cur_channel) * height * width;
    Dtype* cur_top = top_data + 2*index;
    int pixel = 0;
    Dtype max = -FLT_MAX;
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        if (cur_bottom[pixel] > max) {
          max = cur_bottom[pixel];
          cur_top[0] = h;
          cur_top[1] = w;
        }
        pixel++;
      }
    }
  }
}


template <typename Dtype>
void MaxMapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//  this->Forward_cpu(bottom, top);

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count()/2;
  // NOLINT_NEXT_LINE(whitespace/operators)
  MaxMapForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width(), top_data);

  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void MaxMapLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  // Do not backward
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(MaxMapLayer);
#endif

INSTANTIATE_LAYER_GPU_FUNCS(MaxMapLayer);

}  // namespace caffe
