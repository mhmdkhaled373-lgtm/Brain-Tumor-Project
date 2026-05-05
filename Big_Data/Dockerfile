# ── Dockerfile ─────────────────────────────────────────
# Base image: Python 3.13 slim (matches development environment)
FROM python:3.13-slim

# Set working directory inside container
WORKDIR /app

# Copy project files into container
COPY . /app

# Install exact library versions from requirements.txt
# This guarantees same environment as development machine
RUN pip install --no-cache-dir -r requirements.txt

# Expose Jupyter Notebook port
EXPOSE 8888

# Run Jupyter Notebook on container start
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--allow-root", \
     "--NotebookApp.token=''", "--NotebookApp.password=''"]
