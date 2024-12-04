# QSARLit

![Python 3.8](https://img.shields.io/badge/python-3.8-blue) ![Python 3.9](https://img.shields.io/badge/python-3.9-blue) ![Python 3.10](https://img.shields.io/badge/python-3.10-blue) ![Docker](https://img.shields.io/badge/docker-supported-brightgreen)

## Test and Deploy

To test and deploy the project, follow the steps below:

1. Clone the Git repository to your local computer.
2. Make sure you have Docker and Docker Compose installed.
3. In the terminal, navigate to the project's root folder.
4. To start the server in debug mode, execute the following commands:

   ```
   docker compose up --build debugger
   docker compose up debugger
   ```

5. To start the server in production mode, execute the following commands:

   ```
   docker compose up --build production
   docker compose up production
   ```

The above commands will build and start the Docker container for the application.

The application will be exposed at `localhost:8501`.

```console
docker exec -it qsartlit-app bash
```

Make sure to access the application in your browser using the address `localhost:8501`.