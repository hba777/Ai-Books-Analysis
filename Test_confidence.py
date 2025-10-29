import pymongo
import os
from dotenv import load_dotenv
import tkinter as tk
from tkinter import ttk, messagebox

# Load environment variables from .env file
load_dotenv()

# --- MongoDB Configuration ---
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_Agent")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_Agent")

# --- GUI Application ---
class AgentConfidenceUpdater(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Agent Confidence Score Updater")
        self.geometry("400x200")
        self.client = None
        self.collection = None
        self.agent_names = []

        self.connect_to_mongo()
        
        # --- FIX: Change 'if self.collection:' to 'if self.collection is not None:' ---
        if self.collection is not None:
            self.get_agent_names()
            self.create_widgets()

    def connect_to_mongo(self):
        """Connects to MongoDB and sets up the collection object."""
        try:
            self.client = pymongo.MongoClient(MONGO_URI)
            db = self.client[MONGO_DB_NAME]
            self.collection = db[MONGO_COLLECTION_NAME]
            print("Successfully connected to MongoDB.")
        except pymongo.errors.ConnectionFailure as e:
            messagebox.showerror("Connection Error", f"MongoDB connection error: {e}")
            self.collection = None
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
            self.collection = None

    def get_agent_names(self):
        """Fetches agent names from MongoDB for the dropdown menu."""
        # --- FIX: Change 'if self.collection:' to 'if self.collection is not None:' ---
        if self.collection is not None:
            self.agent_names = [doc.get("agent_name") for doc in self.collection.find({}, {"agent_name": 1})]
            if not self.agent_names:
                messagebox.showwarning("No Agents Found", "No agents were found in the database. Please add some first.")

    def create_widgets(self):
        """Creates the GUI components."""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Agent Dropdown Menu
        ttk.Label(main_frame, text="Select Agent:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.selected_agent = tk.StringVar(self)
        if self.agent_names:
            self.selected_agent.set(self.agent_names[0])  # Set initial value
        
        agent_menu = ttk.Combobox(main_frame, textvariable=self.selected_agent, values=self.agent_names, state="readonly")
        agent_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Confidence Score Entry
        ttk.Label(main_frame, text="New Confidence Score:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.confidence_entry = ttk.Entry(main_frame)
        self.confidence_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # Save Button
        save_button = ttk.Button(main_frame, text="Save", command=self.update_confidence_score)
        save_button.grid(row=2, column=0, columnspan=2, pady=10)

        main_frame.columnconfigure(1, weight=1)

    def update_confidence_score(self):
        """Updates the confidence score for the selected agent in MongoDB."""
        agent_name = self.selected_agent.get()
        new_score_str = self.confidence_entry.get()

        if not agent_name or not new_score_str:
            messagebox.showwarning("Input Error", "Please select an agent and enter a confidence score.")
            return

        try:
            new_score = int(new_score_str)
            if not 0 <= new_score <= 100:
                messagebox.showwarning("Invalid Score", "Confidence score must be between 0 and 100.")
                return
        except ValueError:
            messagebox.showerror("Invalid Input", "Confidence score must be a valid number.")
            return

        try:
            # Update the document in MongoDB
            # This check is good practice, but not strictly required here if `self.collection` is already guaranteed to be not None.
            if self.collection is not None:
                update_result = self.collection.update_one(
                    {"agent_name": agent_name},
                    {"$set": {"confidence_score": new_score}}
                )

                if update_result.modified_count == 1:
                    messagebox.showinfo("Success", f"Confidence score for '{agent_name}' updated to {new_score}.")
                else:
                    messagebox.showwarning("Update Failed", f"Could not find or update agent '{agent_name}'.")

        except Exception as e:
            messagebox.showerror("Database Error", f"An error occurred while updating the database: {e}")

    def on_closing(self):
        """Closes the MongoDB connection on application exit."""
        if self.client:
            self.client.close()
            print("MongoDB connection closed.")
        self.destroy()

if __name__ == "__main__":
    app = AgentConfidenceUpdater()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()