FROM python:3.10
RUN apt update && apt install -y python3-pip
RUN pip3 install numpy torch opencv-python flask celery
COPY main.py app/
COPY birdResNet.pt app/
COPY class_to_label.json app/
CMD ["python3", "app/main.py" ]