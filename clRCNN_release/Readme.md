1. Odroid ubuntu 16.04 ��ġ (X86�� ��� ubuntu 16.04 + GPU driver + opencl sdk ��ġ)
2. Dependency Library ��ġ 
  apt-get install libffi-dev python-dev python-pip python-numpy 
  apt-get install python-mako python-yaml protobuf-compiler python-tk python-opencv
  pip install pyopencl 
  pip install enum
  pip install protobuf
  pip install easydict
3. protobuf compile
  ./build.sh
4. ����
  Simple demo: python test.py
  MNIST demo: python mnist.py
  Faster-RCNN demo: python demo.py (���� odroid���� ���� �Ұ���, ROI pooling ����)
  
 
 
 