docker pull tensorflow/tensorflow:latest-gpu-py3
nvidia-docker run -it -p 8888:8888 --rm -v $(pwd):/iceberg tensorflow/tensorflow:latest-gpu-py3 bash
