1. Odroid ubuntu 16.04 설치 (X86의 경우 ubuntu 16.04 + GPU driver + opencl sdk 설치)
2. Dependency Library 설치 
  apt-get install libffi-dev python-dev python-pip python-numpy python-mako python-yaml protobuf-compiler libprotobuf-dev libboost-python-dev python-opencv python-matplotlib python-scipy
  pip install pyopencl 
  pip install enum
  pip install protobuf
  pip install easydict
3. protobuf compile
  ./build.sh
4. 실행
  Simple demo: python test.py
  MNIST demo: python mnist.py
  Faster-RCNN demo: python demo.py 
  

Notice. 
1. dependency가 업데이트 되었습니다. 기존 라이브러리 외에 추가된 라이브러리들이 있으니 해당 라이브러리를 설치하셔야 합니다.
2. proto read 성능이 향상되었습니다.
3. 내부 python framework 구현 부분 일부가 변경되었습니다. 
4. Odroid에서 Faster-RCNN 구동이 가능합니다. 단, 배포된 코드 외적인 버그가 존재하여 동일한 shell에서 2번 실행이 불가능합니다. 다시 demo를 구동하려면 다른 shell로 접속하시고 실행시키면 됩니다.
5. py-faster-rcnn에서 제공하는 레퍼런스 네트워크가 함께 제공됩니다. VOC2007 test set에서 59.31 %의 mAP를 가집니다.
  
 
 
 