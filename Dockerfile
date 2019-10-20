FROM python:3.7-slim
RUN pip3 install boto3
RUN pip3 install numpy
RUN pip3 install scikit-learn
RUN pip3 install awscli
RUN mkdir /src
COPY . /src
RUN pip3 install -r /src/requirements.txt
CMD [“python”, “/src/adaboost_model.py”]
