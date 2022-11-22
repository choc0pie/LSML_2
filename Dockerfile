FROM python:3.10
RUN apt update && apt install -y python3-pip
RUN apt install -y libgl1-mesa-glx
RUN pip3 install numpy \
				torch==1.12.1 \
				torchvision==0.13.1 \
				opencv-python \
				flask 
COPY main.py app/
COPY birdResNet.pt app/
COPY class_to_label.json app/
CMD ["python3", "app/main.py" ]