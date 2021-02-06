#include <Python.h>  // NOLINT(build/include_alpha)
#include <boost/python.hpp>
#include <numpy/arrayobject.h>

#include "proto_reader.h"

#include <iostream>
using namespace boost::python;

struct ProtoReaderWrapper
{
  ProtoReader reader_;

  ProtoReaderWrapper(std::string path):reader_(ProtoReader(path)){}

  int numBlobs(std::string name){return reader_.numBlobs(name);}

  boost::python::object getBlob(std::string name, int idx)
  {
    Blob blob = reader_.getBlob(name, idx);
    std::vector<npy_intp> dim(blob.shape.begin(), blob.shape.end());

    PyObject* arr = 
      PyArray_SimpleNewFromData(dim.size(), dim.data(), NPY_FLOAT32, blob.data.get());
    boost::python::handle<> handle(arr);
    return boost::python::object(handle);
  }
};

BOOST_PYTHON_MODULE(proto_reader)
{
  import_array();
  
  class_<ProtoReaderWrapper>("ProtoReader", init<std::string>())
    .def("num_blobs", &ProtoReaderWrapper::numBlobs)
    .def("get_blob", &ProtoReaderWrapper::getBlob);
}

