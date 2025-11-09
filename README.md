LIVE : https://synhack-frontend.onrender.com/

# Synapse: Candidate Scoring and Ranking System

Synapse is a web application designed to streamline the recruitment process by scoring and ranking candidates based on customizable preferences. It features a React-based frontend for an interactive user experience and a powerful Flask backend for data processing and scoring.

## Features

  * **Advanced Filtering:** Filter candidates based on location, years of experience, required skills, and expected CTC (Cost To Company).
  * **Weighted Scoring:** Assign weights to different parameters to tailor the ranking algorithm to your specific needs.
  * **Dynamic Ranking:** Candidates are ranked in real-time based on the selected filters and weights.
  * **Pagination:** Easily navigate through the list of ranked candidates.
  * **Responsive Design:** The application is designed to work seamlessly across a range of devices.

## Backend Setup

The backend is a Flask application that handles candidate data processing and scoring.

### Prerequisites

  * Python 3.x
  * pip (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/aryapratikofficial/synhack.git
    cd synhack/backend
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements_api.txt
    ```

4.  **Generate the dataset:**
    The `dataset_gen.py` script creates a `candidates.csv` file with synthetic candidate data.

    ```bash
    python dataset_gen.py
    ```

5.  **Run the server:**

    ```bash
    flask run
    ```

    The backend API will be available at `http://127.0.0.1:5000`.

## Frontend Setup

The frontend is a React application built with Vite.

### Prerequisites

  * Node.js
  * npm (Node Package Manager)

### Installation

1.  **Navigate to the frontend directory:**

    ```bash
    cd ../frontend
    ```

2.  **Install dependencies:**

    ```bash
    npm install
    ```

3.  **Run the development server:**

    ```bash
    npm run dev
    ```

    The application will be accessible at `http://localhost:5173`.

## API Endpoints

The backend provides the following main endpoint:

  * **`POST /api/candidates/score`**: Scores and ranks candidates based on the provided JSON payload.

    **Example Request:**

    ```json
    {
        "location": "New York",
        "experience": { "min": "3", "max": "7" },
        "ctc_range": { "min": "30", "max": "100" },
        "skills": ["Python", "SQL", "AWS"],
        "weights": {
            "location": 50,
            "experience": 70,
            "skills": 80,
            "ctc_range": 40
        },
        "page": 1,
        "per_page": 20
    }
    ```

## File Descriptions

### Backend

  * **`candidate_api.py`**: The main Flask application file containing the API endpoints.
  * **`dataset_gen.py`**: A script to generate a synthetic dataset of candidates.
  * **`score_candidate.py`**: Contains the logic for scoring candidates based on various parameters.
  * **`requirements_api.txt`**: A list of Python dependencies for the backend.

### Frontend

  * **`App.jsx`**: The main component of the React application.
  * **`App.css`**: CSS styles for the application.
  * **`package.json`**: Lists the frontend dependencies and scripts.

-----

Crafted by **Team Dijkstra** © 2025
