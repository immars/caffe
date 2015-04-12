#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void MaxMapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void MaxMapLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  top[0]->Reshape(bottom[0]->num(), 2 * bottom[0]->channels(), 1, 1);
}

template <typename Dtype>
void MaxMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int width = bottom[0]->width();
  int height = bottom[0]->height();
  // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < bottom[0]->channels(); ++c) {
      int pixel = 0;
      Dtype max = -FLT_MAX;
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          if (bottom_data[pixel] > max) {
            max = bottom_data[pixel];
            top_data[0] = h;
            top_data[1] = w;
          }
          pixel++;
        }
      }
      // compute offset
      bottom_data += bottom[0]->offset(0, 1);
      top_data += 2;
    }
  }
}

template <typename Dtype>
void MaxMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Do not backward
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(MaxMapLayer);
#endif

INSTANTIATE_CLASS(MaxMapLayer);
REGISTER_LAYER_CLASS(MaxMap);

}  // namespace caffe
