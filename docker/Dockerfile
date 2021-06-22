# To build this container, go to ESMValCore root folder and execute :
# docker build -t esmvalcore:latest . -f docker/Dockerfile
FROM bsces/miniconda3:debian-stretch-20190610

WORKDIR /src/ESMValCore
COPY environment.yml .
RUN conda update -n base -y conda && conda install mamba -c conda-forge && mamba env update --name base --file environment.yml && mamba clean --all -y

# Make RUN commands use the new environment:
SHELL ["/bin/bash", "-c"]

COPY . .
RUN pip install . && pip cache purge

ENTRYPOINT ["esmvaltool"]
