#include "proto_reader.h"

#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <cstring>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <typeinfo>

using google::protobuf::io::FileInputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::Message;

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);

  if(fd == -1)
  {
    std::cerr  << "File not found: " << filename << std::endl;
    assert(0);
  }

  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

ProtoReader::ProtoReader(std::string model_path)
{
  caffe::NetParameter proto;
  read_model(model_path, proto);
  map_model(proto);
  std::cout << "Binary proto read successful" << std::endl;
}

void ProtoReader::read_model(std::string model_path, caffe::NetParameter& proto)
{
  bool success = ReadProtoFromBinaryFile(model_path.c_str(), &proto);
  
  if(success == false)
  {
    std::cerr << "Proto read fail..." << std::endl;
    assert(0);
  }
}

void ProtoReader::map_model(caffe::NetParameter& proto)
{
  auto& layer = proto.layer();
  for(auto l = layer.begin(); l != layer.end(); ++l)
  {
    if(l->blobs().size() > 0)
      this->blobs[l->name()] = std::vector<Blob>();

    auto& blob = l->blobs();
    for(auto b = blob.begin(); b != blob.end(); ++b)
    {
      int blob_size = b->data().size();

      std::shared_ptr<float> data(new float[blob_size], [](float* d){delete[] d;});
      memcpy(data.get(), b->data().data(), blob_size * sizeof(float));

      std::vector<int> dim(b->shape().dim().begin(), b->shape().dim().end());

      this->blobs[l->name()].push_back({data, dim});
    }
  }

}

int ProtoReader::numBlobs(std::string name)
{
  auto itr = this->blobs.find(name);
  if(itr == this->blobs.end())
  {
    return -1;
  }
  else
  {
    return this->blobs[name].size();
  }  
}

Blob ProtoReader::getBlob(std::string name, int idx)
{
  auto itr = this->blobs.find(name);
  if(itr == this->blobs.end() || itr->second.size() < idx+1)
  {
    return Blob();
  }
  else
  {
    return itr->second[idx];
  }
}

/*
int main(void)
{
  std::cout <<"test";
  ProtoReader reader("../demo/ZF_faster_rcnn_final.caffemodel");
  
  Blob b = reader.getBlob("conv1", 0);

  std::cout << b.data.get()[0] <<   ' ' << b.data.get()[1] <<std::endl;
  
  int a;
  std::cin >> a;  
}
*/
