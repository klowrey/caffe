#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

// set static callback member
//template <typename Dtype>
//typename GenericMemoryDataLayer<Dtype>::DataDimCallback 
//	GenericMemoryDataLayer<Dtype>::data_dim_callback_ = NULL;

template <typename Dtype>
void GenericMemoryDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  batch_size_ = this->layer_param_.generic_memory_data_param().batch_size();
  channels_ = this->layer_param_.generic_memory_data_param().channels();
  height_ = this->layer_param_.generic_memory_data_param().height();
  width_ = this->layer_param_.generic_memory_data_param().width();
  size_ = channels_ * height_ * width_;
  CHECK_GT(batch_size_, 0) <<
      "batch_size, must be specified and positive in generic_memory_data_param";

  top[0]->Reshape(batch_size_, channels_, height_, width_);
  data_ = NULL;

  /*
  CHECK(data_dim_callback_);
  data_dim_callback_(this->layer_param_.name(), dim_);
  CHECK_EQ(dim_.size(), top.size()) << "number of inputs must be the same as number of top blobs";
  data_.resize(top.size());
    
  for(size_t i=0; i<top.size(); i++) {
  	data_[i] = NULL;
    top[i]->Reshape(batch_size_, dim_[i], 1, 1);
  }
  */
}

template <typename Dtype>
void GenericMemoryDataLayer<Dtype>::Reset(Dtype* data, int n) {
  CHECK(data);
  CHECK_EQ(n % batch_size_, 0) << "n must be a multiple of batch size";
  /*
  for(size_t i=0; i<data_.size(); i++) {
	data_[i] = data[i];
  }
  */
  data_ = data;
  n_ = n;
  pos_ = 0;
}

template <typename Dtype>
void GenericMemoryDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(data_) << "GenericMemoryDataLayer needs to be initalized by calling Reset";
	  
  top[0]->Reshape(batch_size_, channels_, height_, width_);
  top[0]->set_cpu_data(data_ + pos_ * size_);
  pos_ = (pos_ + batch_size_) % n_;
}

INSTANTIATE_CLASS(GenericMemoryDataLayer);
REGISTER_LAYER_CLASS(GenericMemoryData);

}  // namespace caffe
