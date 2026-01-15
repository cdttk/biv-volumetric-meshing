FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y wget libglib2.0-0 libgl1 libxrender1 libx11-6

ENV VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN rm requirements.txt

RUN wget https://git.opencarp.org/api/v4/projects/16/packages/generic/opencarp-appimage/v11.0/openCARP-v11.0-x86_64_AppImage.tar.gz
RUN tar xf openCARP-v11.0-x86_64_AppImage.tar.gz
RUN ./openCARP-v11.0-x86_64_AppImage/openCARP-v11.0-x86_64.AppImage --appimage-extract
RUN mv squashfs-root /opt/openCARP
RUN rm -rf openCARP-v11.0-x86_64_AppImage*

WORKDIR src
COPY src/ .

ENTRYPOINT ["python3", "run.py", "--data-dir", "/data", "--carp-bin-dir", "/opt/openCARP/usr/bin", "--workspace-dir", "/nnunet"]
