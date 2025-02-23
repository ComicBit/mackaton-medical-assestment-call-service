# InternistAI

This repository implements **InternistAI**, an AI-driven solution designed to streamline the medical visit process by conducting initial evaluations, matching patient symptoms with verified medical data, and connecting users with the right specialist. The goal is to reduce unnecessary visits, prioritize urgent cases, and help address the growing doctor shortage.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution Overview](#solution-overview)
- [Value Proposition](#value-proposition)
- [Tech Stack](#tech-stack)
- [File Structure & Code Explanation](#file-structure--code-explanation)
- [Usage Instructions](#usage-instructions)
- [API Endpoint](#api-endpoint)
- [Repository & Dataset Links](#repository--dataset-links)

---

## Problem Statement

Every day, **500,000 patients** visit internists in the US. With an anticipated shortage of up to **64,000 physicians** by the end of 2024 (according to McKinsey), the healthcare system is under immense pressure. The challenge is to improve efficiency and patient outcomes by reducing unnecessary doctor visits and ensuring that critical cases receive prompt attention.

---

## Solution Overview

**InternistAI** addresses these challenges by:
- Conducting initial evaluations using a TF-IDF-based algorithm.
- Matching patient symptoms against a verified dataset (HDSN).
- Recommending the most likely diseases and suggesting the next best symptom to check.
- Automating appointment bookings (synthetically generated for demonstration).
- Providing a summary of the conversation for record-keeping.

The system minimizes human error and continuously learns from new medical research, ensuring that patients are directed to the right specialist quickly.

---

## Value Proposition

- **Right Doctor, First Time:** Intelligent symptom assessment reduces unnecessary visits by guiding patients to the appropriate specialist.
- **Faster Access to Care:** Prioritizes urgent cases and provides accurate self-care recommendations.
- **Scalable & Efficient:** Automates initial assessments, alleviating pressure on the healthcare system amid a physician shortage.
- **Globally Inclusive:** Caters to both advanced and underserved healthcare markets.
- **Continuous Learning:** The AI evolves with ongoing medical research, ensuring improved accuracy over time.

---

## Tech Stack

- **Flask:** Provides a lightweight web server to expose a single webhook endpoint for tool interactions.
- **Python Standard Libraries:** Utilized for data handling (`csv`, `json`, `datetime`, etc.), mathematical computations, and file operations.
- **Elevenlabs Agent Tool:** Integrates with our Flask server to enable a grounded medical assessment based on the HDSN dataset.
- **HDSN Dataset:** A verified dataset used to calculate TF-IDF values for symptoms and diseases, reducing the risk of hallucinations in diagnosis.

---

## File Structure & Code Explanation

### 1. Tool Class Definition
- **Purpose:** Defines a reusable `Tool` class which wraps a function along with its name, description, expected arguments, and outputs.
- **Usage:** Each tool (e.g., symptom listing, diagnosis, appointments, summary saving) is instantiated as an object of this class.

### 2. Data Structures & TF-IDF Diagnosis
- **Data Loading:** Reads symptom-disease data from a TSV file (`symptoms-DO.tsv`), computes TF-IDF values, and precomputes L2 norms for each disease.
- **Diagnosis Logic:** Implements a cosine-like similarity calculation based on present symptoms to rank diseases and suggest the next best symptom to inquire about.

### 3. Additional Tools
- **Appointments Generation:** Creates synthetic appointment slots for the next 2 days.
- **Summary Saving:** Saves a conversation summary as a JSON file with a timestamp for unique identification.

### 4. Tool Object Wrappers
- **Tools Provided:** 
  - `list_all_symptoms`: Lists all known symptoms.
  - `diagnose_symptoms`: Returns possible diseases and next symptom suggestions based on input symptoms.
  - `available_appointments`: Provides synthetic available appointment slots.
  - `save_summary`: Saves a conversation summary locally.

### 5. Flask Web App
- **Webhook Endpoint:** A single endpoint (`/webhook/tools`) accepts POST requests. The request must specify the `tool_name` and a JSON object of `arguments`.
- **Request Handling:** The endpoint validates the input and calls the corresponding tool function, returning a JSON response.

### 6. Application Entry Point
- **Data Loading:** Automatically loads the dataset when the server starts.
- **Server Run:** The Flask server is configured to run on port 10000 in debug mode.

---

## Usage Instructions

### Prerequisites

- **Python 3.x** installed on your system.
- Required Python libraries (Flask, etc.). You can install dependencies using:
  ```bash
  pip install flask
Running the Application
Clone the Repository:

bash
Copy
git clone https://github.com/ComicBit/mackaton-medical-assestment-call-service.git
cd mackaton-medical-assestment-call-service
Ensure Data Availability:

Verify that the symptoms-DO.tsv file is in the same directory as the main Python file.
Start the Flask Server:

bash
Copy
python <filename>.py
Replace <filename>.py with the actual name of the main file.

Interact with the Webhook:

Send a POST request to http://localhost:10000/webhook/tools with the desired tool name and arguments in JSON format.
Example using diagnose_symptoms:
json
Copy
{
  "tool_name": "diagnose_symptoms",
  "arguments": {
    "symptom_dict": {
      "fever": 1,
      "cough": 1
    }
  }
}
API Endpoint
URL: /webhook/tools
Method: POST
Request JSON Format:
json
Copy
{
  "tool_name": "name_of_tool",
  "arguments": {
    // corresponding arguments for the tool
  }
}
Response: JSON output as defined by the specific tool’s functionality (diagnosis results, available appointments, or confirmation of summary save).
Repository & Dataset Links
Repository: ComicBit/mackaton-medical-assestment-call-service
Dataset: HDSN Dataset on GitHub
