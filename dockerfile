FROM python:3.8-alpine
RUN apk add --update --no-cache py3-numpy
RUN python -m pip install --upgrade pip
RUN pip install dwave-ocean-sdk==2.4.0
ENV PYTHONPATH=/usr/lib/python3.8/site-packages
COPY main.py .
CMD ["python", "main.py"]
