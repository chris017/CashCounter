# Use the base image from the devcontainer configuration
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Set the working directory
WORKDIR /app

# Copy the package files to the container
COPY packages.txt packages.txt
COPY requirements.txt requirements.txt
COPY . .

# Install OS packages
RUN if [ -f packages.txt ]; then \
        apt update && \
        apt upgrade -y && \
        xargs apt install -y < packages.txt; \
    fi

# Install Python packages
RUN if [ -f requirements.txt ]; then \
        pip3 install --user -r requirements.txt && \
        pip3 install -q git+https://github.com/THU-MIG/yolov10.git; \
    fi

# Install Streamlit
RUN pip3 install --user streamlit

# Print a message indicating that packages have been installed
RUN echo '✅ Packages installed and Requirements met'

# Expose the Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]