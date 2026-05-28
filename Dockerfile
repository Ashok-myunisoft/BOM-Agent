# Use an official lightweight Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files into the container
COPY . .

# Expose the port the app runs on
EXPOSE 8007

# Command to run the application using uvicorn on port 8007
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8007"]