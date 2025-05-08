# Use a base image with CUDA 12.2 and Python 3.8
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

LABEL maintainer="Masud Ahmed <mahmed10@umbc.edu>"

# Set environment variable to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.8 and necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    ffmpeg \
    git \
    libsm6 \
    libxext6 \
    libgl1 \
    ninja-build \
    wget \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Update pip and install required Python packages
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.8 && \
    pip install --upgrade pip

# Install the specified Python libraries
RUN pip --no-cache-dir install \
    addict==2.4.0 \
    aiofiles==22.1.0 \
    aiosqlite==0.19.0 \
    ansi2html==1.8.0 \
    anyio==3.7.1 \
    appdirs==1.4.4 \
    argon2-cffi==23.1.0 \
    argon2-cffi-bindings==21.2.0 \
    arrow==1.2.3 \
    async-lru==2.0.4 \
    babel==2.12.1 \
    beautifulsoup4==4.12.2 \
    bleach==6.0.0 \
    comm==0.1.4 \
    configargparse==1.7 \
    dash==2.12.1 \
    dash-core-components==2.0.0 \
    dash-html-components==2.0.0 \
    dash-table==5.0.0 \
    debugpy==1.6.7.post1 \
    defusedxml==0.7.1 \
    deprecation==2.1.0 \
    descartes==1.1.0 \
    docker-pycreds==0.4.0 \
    docopt==0.6.2 \
    dominate==2.8.0 \
    einops==0.6.1 \
    entrypoints==0.4 \
    exceptiongroup==1.1.3 \
    fancycompleter==0.9.1 \
    fastjsonschema==2.18.0 \
    filelock==3.13.1 \
    fire==0.5.0 \
    flask==2.2.5 \
    fqdn==1.5.1 \
    fsspec==2024.2.0 \
    gitdb==4.0.11 \
    gitpython==3.1.42 \
    google-auth-oauthlib==1.0.0 \
    grpcio==1.57.0 \
    huggingface-hub==0.20.3 \
    importlib-resources==6.0.1 \
    ipykernel==6.25.1 \
    ipython-genutils==0.2.0 \
    ipywidgets==8.1.0 \
    isoduration==20.11.0 \
    itsdangerous==2.1.2 \
    jinja2==3.1.2 \
    joblib==1.3.2 \
    json5==0.9.14 \
    jsonpointer==2.4 \
    jsonschema==4.19.0 \
    jsonschema-specifications==2023.7.1 \
    jupyter==1.0.0 \
    jupyter-client==7.4.9 \
    jupyter-console==6.6.3 \
    jupyter-core==5.3.1 \
    jupyter-events==0.7.0 \
    jupyter-lsp==2.2.0 \
    jupyter-packaging==0.12.3 \
    jupyter-server==2.7.1 \
    jupyter-server-fileid==0.9.0 \
    jupyter-server-terminals==0.4.4 \
    jupyter-server-ydoc==0.8.0 \
    jupyter-ydoc==0.2.5 \
    jupyterlab==3.6.5 \
    jupyterlab-pygments==0.2.2 \
    jupyterlab-server==2.24.0 \
    jupyterlab-widgets==3.0.8 \
    llvmlite==0.40.1 \
    markupsafe==2.1.3 \
    matplotlib==3.5.2 \
    mistune==3.0.1 \
    nbclassic==1.0.0 \
    nbclient==0.8.0 \
    nbconvert==7.7.4 \
    nbformat==5.7.0 \
    nest-asyncio==1.5.7 \
    notebook==6.5.5 \
    notebook-shim==0.2.3 \
    numba==0.57.1 \
    nuscenes-devkit==1.1.10 \
    open3d==0.15.2 \
    opencv-python==4.8.0.76 \
    overrides==7.4.0 \
    pandas==2.0.3 \
    pandocfilters==1.5.0 \
    pdbpp==0.10.3 \
    pipreqs==0.4.13 \
    pkgutil-resolve-name==1.3.10 \
    platformdirs==3.10.0 \
    plotly==5.16.1 \
    prometheus-client==0.17.1 \
    protobuf==4.24.0 \
    psutil==5.9.5 \
    pycocotools==2.0.7 \
    pyquaternion==0.9.9 \
    pyrepl==0.9.0 \
    python-json-logger==2.0.7 \
    pytz==2023.3 \
    pyyaml==6.0.1 \
    pyzmq==24.0.1 \
    qtconsole==5.4.3 \
    qtpy==2.3.1 \
    referencing==0.30.2 \
    regex==2023.12.25 \
    retrying==1.3.4 \
    rfc3339-validator==0.1.4 \
    rfc3986-validator==0.1.1 \
    rpds-py==0.9.2 \
    safetensors==0.4.2 \
    scikit-learn==1.3.0 \
    scipy==1.10.1 \
    send2trash==1.8.2 \
    sentry-sdk==1.40.5 \
    setproctitle==1.3.3 \
    shapely==1.8.5 \
    smmap==5.0.1 \
    sniffio==1.3.0 \
    soupsieve==2.4.1 \
    tenacity==8.2.3 \
    tensorboard==2.14.0 \
    tensorboard-data-server==0.7.1 \
    termcolor==2.3.0 \
    terminado==0.17.1 \
    threadpoolctl==3.2.0 \
    tinycss2==1.2.1 \
    tokenizers==0.15.2 \
    tomli==2.0.1 \
    tomlkit==0.12.1 \
    transformers==4.38.1 \
    tzdata==2023.3 \
    uri-template==1.3.0 \
    wandb==0.16.3 \
    webcolors==1.13 \
    webencodings==0.5.1 \
    websocket-client==1.6.1 \
    werkzeug==2.2.3 \
    widgetsnbextension==4.0.8 \
    wmctrl==0.5 \
    y-py==0.6.0 \
    yarg==0.1.9 \
    ypy-websocket==0.8.4 \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    torch-fidelity==0.3.0 \
    timm==1.0.15 \
    pytorch-lightning==2.4.0


# Create a symbolic link for python3 to python
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Expose the port for JupyterLab and code-server
EXPOSE 8443

# Start JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--notebook-dir=/notebooks"]
