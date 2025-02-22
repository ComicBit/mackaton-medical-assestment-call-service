import csv
import math
import os
import signal
from collections import defaultdict
from flask import Flask, request, jsonify
import datetime
import random
import json  # needed to handle JSON serialization

##############################################################################
# 1) The Tool class definition (as provided)
##############################################################################

class Tool:
    """
    A class representing a reusable piece of code (Tool).
    
    Attributes:
        name (str): Name of the tool.
        description (str): A textual description of what the tool does.
        func (callable): The function this tool wraps.
        arguments (list): A list of argument (name, type) pairs.
        outputs (str): A textual or structured description of the return value(s).
    """
    def __init__(self, 
                 name: str, 
                 description: str, 
                 func: callable, 
                 arguments: list,
                 outputs: str):
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.outputs = outputs

    def to_string(self) -> str:
        args_str = ", ".join([
            f"{arg_name}: {arg_type}" for arg_name, arg_type in self.arguments
        ])
        return (
            f"Tool Name: {self.name}, "
            f"Description: {self.description}, "
            f"Arguments: {args_str}, "
            f"Outputs: {self.outputs}"
        )

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


##############################################################################
# 2) Existing data structures and functions for diagnosis (unchanged)
##############################################################################

disease_counts = defaultdict(int)
disease_symptom_counts = defaultdict(lambda: defaultdict(int))
disease_symptom_probs = defaultdict(dict)
all_symptoms = set()
all_diseases_list = []
total_cooccurs = 0

DATA_FILE = os.path.join(os.path.dirname(__file__), "symptoms-DO.tsv")

def load_data():
    global total_cooccurs

    print(f"Loading data from {DATA_FILE} ...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Convert symptom and disease names to lowercase
            symptom = row["symptom_name"].strip().lower()
            disease = row["disease_name"].strip().lower()  # if needed
            try:
                cooccurs = int(row["cooccurs"].strip())
            except:
                cooccurs = 0

            all_symptoms.add(symptom)
            disease_symptom_counts[disease][symptom] += cooccurs
            disease_counts[disease] += cooccurs
            total_cooccurs += cooccurs

            if disease not in all_diseases_list:
                all_diseases_list.append(disease)

    print(f"Loaded {len(all_diseases_list)} diseases and {len(all_symptoms)} unique symptoms.")
    print("Computing probabilities with Laplace smoothing...")

    num_symptoms = len(all_symptoms)
    for disease, dcount in disease_counts.items():
        for symptom in all_symptoms:
            count = disease_symptom_counts[disease][symptom]
            prob = (count + 1.0) / (dcount + num_symptoms)
            disease_symptom_probs[disease][symptom] = prob

    print("Data loading completed.")

def compute_disease_probability(symptom_dict):
    results = []
    total_disease_occurrences = sum(disease_counts.values())
    if total_disease_occurrences == 0:
        return []

    for disease, count in disease_counts.items():
        prior_prob = count / total_disease_occurrences
        import math
        logp = math.log(prior_prob + 1e-9)
        for s, val in symptom_dict.items():
            p_s_given_d = disease_symptom_probs[disease].get(s, 1e-9)
            if val == 1:
                logp += math.log(p_s_given_d + 1e-9)
            else:
                logp += math.log((1 - p_s_given_d) + 1e-9)
        results.append((disease, logp))

    max_logp = max(r[1] for r in results)
    exp_probs = []
    for (d, lp) in results:
        exp_probs.append((d, math.exp(lp - max_logp)))
    sum_exp = sum(x[1] for x in exp_probs)
    final = []
    for (d, ep) in exp_probs:
        prob = ep / (sum_exp if sum_exp else 1e-9)
        final.append((d, prob))

    final.sort(key=lambda x: x[1], reverse=True)
    return final

def suggest_next_symptom(symptom_dict):
    import math
    if len(symptom_dict) == len(all_symptoms):
        return None

    top_diseases = compute_disease_probability(symptom_dict)[:5]
    if not top_diseases:
        return None

    top_disease_names = [d for (d, _p) in top_diseases]
    candidates = [s for s in all_symptoms if s not in symptom_dict]

    best_symptom = None
    best_variance = -1
    for s in candidates:
        values = [disease_symptom_probs[d].get(s, 0.0) for d in top_disease_names]
        if not values:
            continue
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        if var > best_variance:
            best_variance = var
            best_symptom = s

    return best_symptom

def list_all_symptoms_logic():
    return {"all_symptoms": sorted(all_symptoms)}

def diagnose_symptoms_logic(symptom_dict):
    # Normalize input keys to lowercase
    symptom_dict = {k.lower(): v for k, v in symptom_dict.items()}
    ranked = compute_disease_probability(symptom_dict)
    top_5 = []
    for d, p in ranked[:5]:
        top_5.append({"disease": d, "probability": p})
    suggestion = suggest_next_symptom(symptom_dict)
    return {
        "possible_diseases": top_5,
        "next_symptom_suggestions": [suggestion] if suggestion else []
    }

##############################################################################
# 3) New Tool: Synthetic Appointment Slot Finder
##############################################################################

def available_appointments_logic():
    """
    Generates synthetic appointment slots for the next 2 days.
    Each appointment is 30 minutes long and randomly chosen within work hours (9:00-17:00).
    Returns a dict mapping each date (YYYY-MM-DD) to a list of 2 available time slots.
    """
    appointments = {}
    start_hour = 9
    end_hour = 17  # appointments can start as late as 16:30

    for day in range(1, 3):
        date = (datetime.date.today() + datetime.timedelta(days=day)).isoformat()
        possible_slots = []
        current_time = datetime.datetime(2000, 1, 1, start_hour, 0)
        while current_time.hour < end_hour or (current_time.hour == end_hour and current_time.minute == 0):
            possible_slots.append(current_time.strftime("%H:%M"))
            current_time += datetime.timedelta(minutes=30)
            if current_time.hour == end_hour and current_time.minute > 0:
                break

        chosen_slots = random.sample(possible_slots, 2)
        appointments[date] = sorted(chosen_slots)
    return {"appointments": appointments}

##############################################################################
# 4) New Tool: Save Summary and Transcript to Local File
##############################################################################

def save_summary_logic(summary):
    """
    Accepts a summary (JSON serializable data) and saves it to a file locally.
    The file is named with a timestamp for uniqueness (e.g., summary_20250223T153000.json).
    
    Parameters:
        summary (dict): A JSON serializable object containing the conversation summary and transcript.
        
    Returns:
        dict: A confirmation message with the filename.
    """
    # Create a timestamp string in the format YYYYMMDDTHHMMSS
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = f"summary_{timestamp}.json"
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        return {"message": "Summary saved successfully", "filename": filename}
    except Exception as e:
        return {"error": str(e)}

##############################################################################
# 5) Wrap functions into Tool objects
##############################################################################

list_symptoms_tool = Tool(
    name="list_all_symptoms",
    description="Returns a list of all known symptoms in the database.",
    func=list_all_symptoms_logic,
    arguments=[],
    outputs="JSON with 'all_symptoms' array"
)

diagnose_tool = Tool(
    name="diagnose_symptoms",
    description="Given user symptom presence, returns likely diseases plus next symptom suggestions.",
    func=diagnose_symptoms_logic,
    arguments=[("symptom_dict", "dict")],
    outputs="JSON with 'possible_diseases' and 'next_symptom_suggestions'"
)

appointments_tool = Tool(
    name="available_appointments",
    description=("Returns available synthetic appointment slots for the next 2 days. "
                 "Each appointment is 30 minutes long, randomly generated within work hours (9:00-17:00)."),
    func=available_appointments_logic,
    arguments=[],
    outputs="JSON with 'appointments' mapping dates (YYYY-MM-DD) to a list of time slots"
)

save_summary_tool = Tool(
    name="save_summary",
    description=("Saves a conversation summary (or transcript) to a local JSON file with a timestamp. "
                 "The summary must be a JSON serializable object."),
    func=save_summary_logic,
    arguments=[("summary", "dict")],
    outputs="JSON with a confirmation message and the filename where the summary was saved"
)

ALL_TOOLS = {
    list_symptoms_tool.name: list_symptoms_tool,
    diagnose_tool.name: diagnose_tool,
    appointments_tool.name: appointments_tool,
    save_summary_tool.name: save_summary_tool
}

##############################################################################
# 6) Flask Web App: Single webhook endpoint
##############################################################################

app = Flask(__name__)

@app.route("/webhook/tools", methods=["POST"])
def handle_tool_webhook():
    """
    Expects JSON like:
      {
        "tool_name": "save_summary",
        "arguments": {
            "summary": { ... }
        }
      }
    or for other tools:
      {
        "tool_name": "available_appointments",
        "arguments": {}
      }
    """
    data = request.get_json(silent=True) or {}
    tool_name = data.get("tool_name")
    arguments = data.get("arguments", {})

    if not tool_name:
        return jsonify({"error": "No tool_name provided"}), 400
    if tool_name not in ALL_TOOLS:
        return jsonify({"error": f"Unknown tool: {tool_name}"}), 400

    tool = ALL_TOOLS[tool_name]
    try:
        if tool_name == "diagnose_symptoms":
            symptom_dict = arguments.get("symptom_dict", {})
            result = tool(symptom_dict)
        elif tool_name == "list_all_symptoms":
            result = tool()
        elif tool_name == "available_appointments":
            result = tool()
        elif tool_name == "save_summary":
            summary = arguments.get("summary", {})
            result = tool(summary)
        else:
            result = tool(**arguments)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

##############################################################################
# 7) App Entry Point
##############################################################################

if __name__ == "__main__":
    load_data()  # load data for diagnosis tools
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=False, host="0.0.0.0", port=port)
