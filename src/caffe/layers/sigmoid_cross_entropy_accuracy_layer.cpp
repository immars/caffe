#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidCrossEntropyAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_Accuracy layer inputs must have the same count.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidCrossEntropyAccuracyLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
  int total = bottom[0]->count();
  int count = 0;
  Dtype n1 = 0, n0 = 0; // target
  Dtype x1 = 0, x0 = 0; // correct (target same as predict)
  Dtype y1 = 0, y0 = 0; // predict
  for (int i = 0; i < total; ++i) {
    if (bottom_label[i] == -1) {
      continue;
    }
    count++;
    n1 += bottom_label[i] == 1;
    n0 += bottom_label[i] == 0;
    y1 += sigmoid_output_data[i] >= 0.5;
    y0 += sigmoid_output_data[i] < 0.5;
    x1 += (bottom_label[i] == 1) * (sigmoid_output_data[i] >= 0.5)
            * (sigmoid_output_data[i]);
    x0 += (bottom_label[i] == 0) * (sigmoid_output_data[i] < 0.5)
            * (1 - sigmoid_output_data[i]);
//    accuracy += (bottom_label[i] == 0) * (sigmoid_output_data[i] < 0.5)
//        * (0.5 - sigmoid_output_data[i]) * 2
//        + (bottom_label[i] == 1) * (sigmoid_output_data[i] >= 0.5)
//        * (sigmoid_output_data[i] - 0.5) * 2;
  }
  Dtype w1 = 0,w0 = 0;
  Dtype acc1 = 0, acc0 = 0;
  Dtype recall1 = 0, recall0 = 0;
  w1 = n1==0? 0 : -log(n1/count);
  w0 = n0==0? 0 : -log(n0/count);
  acc1 = n1==0? 0 : x1 / n1;
  acc0 = n0==0? 0 : x0 / n0;
  recall1 = y1==0? 0: x1 / y1;
  recall0 = y0==0? 0: x0 / y0;

  Dtype out = w1 / (w1 + w0) * acc1 * recall1 + w0 / (w1 + w0) * acc0 * recall0;
//  LOG(INFO) << "a\t" << accuracy << "\tx1\t" << x1 << "\tx0\t" << x0
//      << "\tcount\t" << count << "\tn1\t" << n1 << "\tn0\t" << n0
//      << "\taccuracy\t" << out << "\tw1\t" << w1 << "\tw0\t" << w0;
  top[0]->mutable_cpu_data()[0] = out;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(SigmoidCrossEntropyAccuracyLayer);
REGISTER_LAYER_CLASS(SigmoidCrossEntropyAccuracy);

}  // namespace caffe
