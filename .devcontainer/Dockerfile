FROM tensorflow/tensorflow:2.13.0-gpu-jupyter

ARG USERNAME
ARG USER_UID
ARG USER_GID

# Create non-root user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Switch to non-root user
USER $USERNAME

# Copy requirements.txt
COPY requirements.txt /tmp/pip/requirements.txt

# Install python packages
RUN pip install -r /tmp/pip/requirements.txt \
    && sudo rm -rf /tmp/pip/

RUN sudo apt-get install -y vim \
    && pip install --user pipx \
    && ~/.local/bin/pipx ensurepath \
    && echo eval "\$(register-python-argcomplete pipx)" >> ~/.bashrc
