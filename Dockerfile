FROM nvcr.io/nvidia/pytorch:20.12-py3
ENV DEBIAN_FRONTEND noninteractive

RUN pip install dlib
RUN pip install gdown
RUN pip install -U scikit-image

RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install -U opencv-python


# WORKDIR /workspace/
# COPY . .

ENTRYPOINT [ "python", "main.py" ]
