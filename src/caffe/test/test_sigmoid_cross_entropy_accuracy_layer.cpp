#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class SigmoidCrossEntropyAccuracyLayerTest : public ::testing::Test {
 protected:
  SigmoidCrossEntropyAccuracyLayerTest()
      : blob_bottom_data_(new Blob<Dtype>()),
        blob_bottom_label_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        top_k_(3) {
    vector<int> shape(2);
    shape[0] = 100;
    shape[1] = 10;
    blob_bottom_data_->Reshape(shape);
    blob_bottom_label_->Reshape(shape);
    FillBottoms();

    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual void FillBottoms() {
    // fill the probability values
    FillerParameter filler_param;
    filler_param.set_max(5);
    filler_param.set_min(5);
    filler_param.set_mean(0);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);

//    const unsigned int prefetch_rng_seed = caffe_rng_rand();
//    shared_ptr<Caffe::RNG> rng(new Caffe::RNG(prefetch_rng_seed));
//    caffe::rng_t* prefetch_rng =
//          static_cast<caffe::rng_t*>(rng->generator());
    Dtype* label_data = blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      label_data[i] = i % 2;
    }
  }

  virtual ~SigmoidCrossEntropyAccuracyLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  int top_k_;
};

TYPED_TEST_CASE(SigmoidCrossEntropyAccuracyLayerTest, TestDtypes);

TYPED_TEST(SigmoidCrossEntropyAccuracyLayerTest, TestSetup) {
  LayerParameter layer_param;
  SigmoidCrossEntropyAccuracyLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(SigmoidCrossEntropyAccuracyLayerTest, TestForwardCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  SigmoidCrossEntropyAccuracyLayer<TypeParam> layer(layer_param);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_LT(this->blob_top_->data_at(0, 0, 0, 0), 0.5);

  TypeParam* label_data = this->blob_bottom_label_->mutable_cpu_data();
  for (int i = 0; i < this->blob_bottom_label_->count(); ++i) {
    label_data[i] = (i % 4) == 0 ? 1 : 0;
  }
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_LT(this->blob_top_->data_at(0, 0, 0, 0), 0.5);

  FillerParameter filler_param;
  filler_param.set_max(1);
  filler_param.set_min(-3);
  filler_param.set_mean(-1);
  UniformFiller<TypeParam> filler(filler_param);
  filler.Fill(this->blob_bottom_data_);

  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_LT(this->blob_top_->data_at(0, 0, 0, 0), 0.625);
}

}  // namespace caffe
