#ifndef _PROTO_READER_H_
#define _PROTO_READER_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "caffe.pb.h"

#include <google/protobuf/text_format.h>

struct Blob
{
  std::shared_ptr<float> data;
  std::vector<int> shape;

  Blob():data(NULL){}
  Blob(std::shared_ptr<float> data, std::vector<int> shape):data(data),shape(shape){}
};

class ProtoReader{
private:  
  std::unordered_map<std::string, std::vector<Blob> > blobs;
  void read_model(std::string model_path, caffe::NetParameter& proto);
  void map_model(caffe::NetParameter& proto);

public:  
  ProtoReader(std::string model_path);
  int numBlobs(std::string name);
  Blob getBlob(std::string name, int idx);
};

#endif