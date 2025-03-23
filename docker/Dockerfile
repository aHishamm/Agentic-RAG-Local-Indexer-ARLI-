FROM python:3.11

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Copy requirements first for better caching
COPY ../requirements.txt /app/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project code
COPY ../src /app/src/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "src/manage.py", "runserver", "0.0.0.0:8000"]