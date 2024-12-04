# Build stage
FROM python:3.9-slim-bookworm as builder

# Create app directory
RUN mkdir /usr/qsarlit/
WORKDIR /usr/qsartlit/

# Add user
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1000 vscode

# Update args in docker-compose.yaml to set the UID/GID of the "vscode" user.
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN if [ "$USER_GID" != "1000" ] || [ "$USER_UID" != "1000" ]; then groupmod --gid $USER_GID vscode && usermod --uid $USER_UID --gid $USER_GID vscode; fi

# Install build dependencies
RUN apt-get -qq update && apt-get install libxrender1 libarchive-dev build-essential git liblapack-dev libblas-dev gfortran libsuitesparse-dev libcairo2 -y \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log 

# Bundle app source
RUN mkdir -p app 
COPY app/* ./app/

# Compile cvxopt
ENV CPPFLAGS="-I/usr/include/suitesparse"

# Install Python dependencies
RUN python3.9 -m pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel
RUN python3.9 -m pip install --no-cache-dir -r app/requirements.txt

# Production stage
FROM builder as production

# Set the working directory in the production stage
WORKDIR /usr/qsarlit/app/

# Set the environment variables
ENV FLASK_APP=/usr/qsarlit/app/api.py
ENV PYTHONPATH "${PYTHONPATH}:/usr/bin/python3.9"

# Expose ports 8501
EXPOSE 8501

# Start supervisor to manage the Flask applications
CMD streamlit run app.py

# Debug stage
FROM builder as debug

# Set the environment variables
ENV FLASK_APP=/usr/predherg/app/api.py
ENV PYTHONPATH "${PYTHONPATH}:/usr/bin/python3.9"

# Install ptvsd for debugging
RUN python3.9 -m pip install --no-cache-dir  debugpy

# Expose ports 5678 and 8501
EXPOSE 5678
EXPOSE 8501

# Start Streamlit app with debugpy for debugging
CMD python3.9 -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m streamlit run app.py --server.port 8501