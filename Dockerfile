FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel AS build
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="America/New_York"
RUN apt-get -q update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mmcv-full==1.6.0

FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="America/New_York"
RUN apt-get -q update && apt-get install -y python3-opencv && rm -rf /var/lib/apt/lists/*
RUN pip install addict yapf opencv-python
COPY --from=build /opt/conda/lib/python3.10/site-packages/mmcv /opt/conda/lib/python3.10/site-packages/mmcv
COPY --from=build /opt/conda/lib/python3.10/site-packages/mmcv_full-1.6.0.dist-info /opt/conda/lib/python3.10/site-packages/mmcv_full-1.6.0.dist-info

WORKDIR /src/EgoHOS
ADD mmseg mmseg
ADD setup.py .
ADD README.md .
ADD egohos/__init__.py egohos/
RUN pip install -e .
ADD egohos egohos

WORKDIR /src
