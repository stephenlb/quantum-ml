FROM python:3.8.2-slim
RUN python -m pip install --upgrade pip
RUN pip install setuptools numpy matplotlib notebook dwave-ocean-sdk==2.4.0
COPY xor-quantum.py .
CMD ["python", "xor-quantum.py"]
