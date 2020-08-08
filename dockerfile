FROM python:3.8.2-slim
RUN python -m pip install --upgrade pip
RUN pip install setuptools numpy matplotlib notebook dwave-ocean-sdk==2.4.0
COPY main.py .
CMD ["python", "main.py"]
