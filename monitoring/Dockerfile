# Start from an official Python base image (e.g., python:3.9-slim).
FROM python:3.9-slim

# Set a working directory inside the container (e.g., /app).
WORKDIR /app

# Copy the requirements.txt file into the container.
COPY requirements.txt /app/

# Install the Python dependencies using pip.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files (app.py) into the container.
COPY . /app

# Expose the port that Streamlit runs on (default is 8501).
EXPOSE 8501

# Define the command (CMD) to run the Streamlit app when the container starts.
CMD ["streamlit", "run", "app.py"]