1. Odroid ubuntu 16.04 설치 (X86의 경우 ubuntu 16.04 + GPU driver + opencl sdk 설치)
2. Dependency Library 설치 
  apt-get install libffi-dev python-dev python-pip python-numpy 
  apt-get install python-mako python-yaml protobuf-compiler python-tk python-opencv
  pip install pyopencl 
  pip install enum
  pip install protobuf
  pip install easydict
3. protobuf compile
  ./build.sh
4. 실행
  Simple demo: python test.py
  MNIST demo: python mnist.py
  Faster-RCNN demo: python demo.py (현재 odroid에서 구동 불가능, ROI pooling 느림)
  
 
 
 