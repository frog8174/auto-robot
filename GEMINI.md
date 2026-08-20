# GEMINI.md

## Project Overview

This project is a Flask web application designed to remotely control a 2-wheel robot and a single relay output. It's intended to run on a Raspberry Pi using the `gpiozero` library for hardware interaction.

The application provides:
*   A web interface with buttons for directional movement (Forward, Backward, Left, Right, Stop).
*   A toggle switch and buttons to control a relay.
*   Keyboard controls for driving the robot (arrow keys for movement, spacebar to stop) and toggling the relay ('R' key).
*   A simple REST API for programmatic control of the robot's movement and the relay's state.

## Building and Running

### Dependencies

The project requires Python 3, Flask, and the `gpiozero` library. You can install the necessary packages using pip:

```bash
pip install Flask gpiozero
```

### Configuration

The GPIO pin assignments and other settings can be configured via environment variables. The following variables are available:

*   `LEFT_IN1`, `LEFT_IN2`: Pins for the left motor.
*   `RIGHT_IN1`, `RIGHT_IN2`: Pins for the right motor.
*   `PWM_SPEED_DEFAULT`: Default motor speed (0.0 to 1.0).
*   `RELAY_PIN`: Pin for the relay.
*   `RELAY_ACTIVE_HIGH`: Set to "1" if the relay is active-high.
*   `HOST`: The host address to bind the server to (defaults to `0.0.0.0`).
*   `PORT`: The port for the web server (defaults to `8000`).
*   `FLASK_DEBUG`: Set to "1" to enable Flask's debug mode.

### Running the Application

To start the web server, run the `app.py` script:

```bash
python app.py
```

The application will be accessible in a web browser at `http://<your-pi-ip>:8000`.

## Development Conventions

*   **Single-File Application:** The entire application, including the web interface, is contained within `app.py`.
*   **Inline Frontend:** The HTML, CSS, and JavaScript for the user interface are embedded directly in the Python script.
*   **REST API:** The backend exposes a simple RESTful API for controlling the robot.
    *   `POST /api/move`: Sends a movement command (`forward`, `backward`, `left`, `right`, `stop`).
    *   `POST /api/relay`: Sets the relay state (`on`, `off`, `toggle`).
    *   `GET /api/status`: Retrieves the current state of the robot.
*   **Graceful Shutdown:** The application is configured to stop all motors and turn off the relay when it exits, ensuring the robot is left in a safe state.
