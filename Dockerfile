FROM fedora:37

ENV MPG /project/mpg
ENV MPGCPP /project/mpgcpp
ENV MPG_PYTHON ${MPG}/wrapper
ENV MPG_NOTEBOOKS /project/notebooks
ENV MPGCPP_BUILD /tmp/build

RUN dnf install -y \
    python3 \
    python-is-python3 \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-devel \
    gcc \
    g++\
    boost \
    boost-devel \
    cmake \
    && dnf clean all

RUN pip3 install --upgrade pip

COPY mpg/requirements.txt ${MPG}/requirements.txt
RUN pip3 install -r ${MPG}/requirements.txt

COPY mpg ${MPG}
COPY mpgcpp ${MPGCPP}

WORKDIR ${MPGCPP}

RUN mkdir -p ${MPGCPP_BUILD}
RUN cmake . -B ${MPGCPP_BUILD}

WORKDIR ${MPGCPP_BUILD}
RUN make
RUN make install

ARG NB_USER="mpg"
ENV JUPYTER_PORT=10053

WORKDIR ${MPG_NOTEBOOKS}
RUN ln -s ${MPG} mpg

RUN useradd -m -s /bin/bash ${NB_USER}
USER ${NB_USER}

COPY notebooks ${MPG_NOTEBOOKS}
WORKDIR ${MPG_NOTEBOOKS}

EXPOSE ${JUPYTER_PORT}

CMD jupyter lab --port=${JUPYTER_PORT} --no-browser --ip=0.0.0.0 --allow-root
