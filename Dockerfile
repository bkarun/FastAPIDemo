# Use an official Python base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy your app code into the container
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
