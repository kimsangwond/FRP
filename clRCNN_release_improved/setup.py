from distutils.core import setup, Extension

# define the directories to search for include files
# to get this to work, you may need to include the path
# to your boost installation. Mine was in 
# '/usr/local/include', hence the corresponding entry.
include_dirs = [ '/usr/local/include']
 
# define the library directories to include any extra
# libraries that may be needed.  The boost::python
# library for me was located in '/usr/local/lib'
library_dirs = [ '/usr/local/lib' ]


setup(
	name = 'proto_reader', 
	version = '0.0.1', 
	ext_modules=[
		Extension('proto_reader',
			sources=['proto/caffe.pb.cc', 'proto/proto_reader.cpp', 'proto/proto_wrapper.cpp'],
			libraries=['protobuf', 'boost_python-py27'],
			include_dirs=include_dirs, library_dirs=library_dirs,
			extra_compile_args=['-std=c++11']
		)
	]
)