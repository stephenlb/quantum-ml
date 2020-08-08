FROM python:3.8.2-slim
RUN python -m pip install --upgrade pip
RUN pip install setuptools numpy matplotlib notebook dwave-ocean-sdk
COPY main.py .
CMD ["python", "main.py"]



#CMD ["/bin/bash"]

#FROM python:3.8-alpine
#RUN apk update
#RUN apk add --update --no-cache py3-numpy
#RUN apk add --virtual build-dependencies alpine-sdk build-base gcc wget git
#RUN pip install dwave-ocean-sdk==2.2.0
#ENV PYTHONPATH=/usr/lib/python3.8/site-packages

