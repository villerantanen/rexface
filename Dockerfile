from ubuntu:16.04

RUN apt-get update
RUN apt-get install -y libboost-all-dev cmake libopenblas-dev liblapack-dev python3-pip git && apt-get clean
RUN cd /tmp && pip3 install setuptools && \
  git clone https://github.com/davisking/dlib.git && cd dlib && \
  mkdir -p build && cd build  && \
  cmake .. -DDLIB_USE_CUDA=0 -DUSE_AVX_INSTRUCTIONS=1 && cmake --build . && \
  cd .. && \
  python3 setup.py install --yes USE_AVX_INSTRUCTIONS --no DLIB_USE_CUDA
RUN pip3 install face_recognition flask scikit-image gunicorn
ADD . /code
WORKDIR /code
EXPOSE 80
CMD ["bash", "run.sh"]



