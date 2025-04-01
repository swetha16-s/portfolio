import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
import joblib
from datetime import datetime
import json
import os

class AdvancedDiseasePredictionSystem:
    def __init__(self):
        self.symptoms = [
            'fever', 'cough', 'loss_of_taste', 'loss_of_smell',
            'body_aches', 'fatigue', 'headache',
            'difficulty_breathing', 'chest_pain',
            'fatigue','weight_loss', 'night_sweats',
            'wheezing','chest_tightness','shortness_of_breath', 'chronic_cough',
            'persistent_cough', 'dizziness','frequent_urination', 'excessive_thirst',
            'weight_change', 'hair_loss', 'irregular_heartbeat','joint_pain', 'swelling', 'stiffness',
            'persistent_sadness', 'loss_of_interest','excessive_worry', 'irregular_heartbeat', 'sweating', 'tremor'
        ]

        self.diseases = [
            'COVID-19', 'Influenza', 'Common Cold', 'Pneumonia',
            'Tuberculosis', 'Asthma', 'COPD',
            'Lung Cancer', 'Heart Disease', 'Diabetes', 'Thyroid Disorder',
            'Arthritis', 'Migraine', 'Depression', 'Anxiety Disorder',
            'Hypertension', 'Stroke', 'Gastroenteritis', 'Food Poisoning',
            'Hepatitis', 'Dengue', 'Malaria', 'Chronic Kidney Disease',
            'Liver Disease', 'Sinusitis',  'Appendicitis',
            'Gastric Ulcer', 'Gallbladder Disease', 'Pancreatitis'
        ]

        self.models = self._initialize_models()
        self.history = []
        self.load_history()

    def _initialize_models(self):
        # Initialize advanced models with optimized parameters
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42
            )
        }
        
        # Generate synthetic training data
        X_train, y_train = self._generate_synthetic_data()
        
        # Train models
        for name, model in models.items():
            model.fit(X_train, y_train)
            
        return models

    def _generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic training data based on medical knowledge"""
        np.random.seed(42)
        X = np.random.rand(n_samples, len(self.symptoms))
        y = np.zeros(n_samples)
        
        # Define disease patterns
        disease_patterns = {
            'COVID-19': ['fever', 'cough', 'loss_of_taste', 'loss_of_smell'],
            'Influenza': ['fever', 'body_aches', 'fatigue', 'headache'],
            'Pneumonia': ['fever', 'cough', 'difficulty_breathing', 'chest_pain'],
            'Tuberculosis': ['cough', 'weight_loss', 'night_sweats', 'fatigue'],
            'Asthma': ['difficulty_breathing', 'wheezing', 'cough', 'chest_tightness'],
            'COPD': ['shortness_of_breath', 'chronic_cough', 'fatigue', 'wheezing'],
            'Lung Cancer': ['persistent_cough', 'chest_pain', 'weight_loss', 'fatigue'],
            'Heart Disease': ['chest_pain', 'shortness_of_breath', 'fatigue', 'dizziness'],
            'Diabetes': ['frequent_urination', 'excessive_thirst', 'fatigue', 'weight_loss'],
            'Thyroid Disorder': ['fatigue', 'weight_change', 'hair_loss', 'irregular_heartbeat'],
            'Arthritis': ['joint_pain', 'swelling', 'stiffness', 'fatigue'],
            'Depression': ['persistent_sadness', 'fatigue', 'loss_of_interest', 'weight_change'],
            'Anxiety Disorder': ['excessive_worry', 'irregular_heartbeat', 'sweating', 'tremor']
        }
        
        # Apply patterns to generate realistic data
        for i in range(n_samples):
            disease_idx = np.random.randint(0, len(self.diseases))
            disease = self.diseases[disease_idx]
            
            if disease in disease_patterns:
                for symptom in disease_patterns[disease]:
                    symptom_idx = self.symptoms.index(symptom)
                    X[i, symptom_idx] = np.random.uniform(0.7, 1.0)
            
            y[i] = disease_idx
            
        return X, y

    def predict(self, symptoms):
        # Convert symptoms to feature vector
        feature_vector = np.zeros(len(self.symptoms))
        for symptom in symptoms:
            if symptom in self.symptoms:
                feature_vector[self.symptoms.index(symptom)] = 1
                
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            proba = model.predict_proba([feature_vector])[0]
            pred_idx = np.argmax(proba)
            confidence = proba[pred_idx]
            
            predictions[name] = {
                'disease': self.diseases[pred_idx],
                'confidence': confidence,
                'probabilities': {
                    self.diseases[i]: float(p)
                    for i, p in enumerate(proba)
                }
            }
            
        # Save prediction to history
        self.save_prediction(symptoms, predictions)
        return predictions

    def save_prediction(self, symptoms, predictions):
        prediction_data = {
            'timestamp': datetime.now().isoformat(),
            'symptoms': symptoms,
            'predictions': predictions
        }
        self.history.append(prediction_data)
        
        # Save to file
        with open('prediction_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)

    def load_history(self):
        try:
            with open('prediction_history.json', 'r') as f:
                self.history = json.load(f)
        except FileNotFoundError:
            self.history = []

class ReportGenerator:
    def __init__(self, predictor, ui):
        self.predictor = predictor
        self.ui = ui
        self.setup_report_tab()
        
    def setup_report_tab(self):
        """Create the Report Generation tab in the UI"""
        self.report_tab = ttk.Frame(self.ui.notebook, padding=20)
        self.ui.notebook.add(self.report_tab, text="Reports")
        
        # Report options
        ttk.Label(
            self.report_tab,
            text="Report Generation",
            style='Header.TLabel'
        ).grid(row=0, column=0, columnspan=2, pady=20)
        
        # Report type selection
        ttk.Label(
            self.report_tab,
            text="Report Type:"
        ).grid(row=1, column=0, sticky='w', padx=5, pady=5)
        
        self.report_type = tk.StringVar(value="Current Diagnosis")
        report_types = ["Current Diagnosis", "History Summary", "Full Patient History"]
        
        report_dropdown = ttk.Combobox(
            self.report_tab,
            textvariable=self.report_type,
            values=report_types,
            state="readonly",
            width=30
        )
        report_dropdown.grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        # Include visualization option
        self.include_viz = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.report_tab,
            text="Include visualizations",
            variable=self.include_viz
        ).grid(row=2, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        # Patient information
        patient_frame = ttk.LabelFrame(self.report_tab, text="Patient Information", padding=10)
        patient_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)
        
        ttk.Label(
            patient_frame,
            text="Patient ID:"
        ).grid(row=0, column=0, sticky='w', padx=5, pady=5)
        
        self.patient_id = tk.StringVar()
        ttk.Entry(
            patient_frame,
            textvariable=self.patient_id,
            width=20
        ).grid(row=0, column=1, sticky='w', padx=5, pady=5)
        
        ttk.Label(
            patient_frame,
            text="Patient Name:"
        ).grid(row=1, column=0, sticky='w', padx=5, pady=5)
        
        self.patient_name = tk.StringVar()
        ttk.Entry(
            patient_frame,
            textvariable=self.patient_name,
            width=30
        ).grid(row=1, column=1, sticky='w', padx=5, pady=5)
        
        # Generate report button
        ttk.Button(
            self.report_tab,
            text="Generate Report",
            command=self.generate_report
        ).grid(row=4, column=0, columnspan=2, pady=20)
        
        # Export options
        export_frame = ttk.LabelFrame(self.report_tab, text="Export Options", padding=10)
        export_frame.grid(row=5, column=0, columnspan=2, sticky="ew", pady=10)
        
        ttk.Button(
            export_frame,
            text="Export as PDF",
            command=self.export_as_pdf
        ).grid(row=0, column=0, padx=5, pady=5)
        
        ttk.Button(
            export_frame,
            text="Export as CSV",
            command=self.export_as_csv
        ).grid(row=0, column=1, padx=5, pady=5)
        
        # Preview area
        preview_frame = ttk.LabelFrame(self.report_tab, text="Report Preview", padding=10)
        preview_frame.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=10)
        
        self.preview_text = tk.Text(preview_frame, wrap=tk.WORD, height=15, width=60)
        self.preview_text.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.preview_text.configure(yscrollcommand=scrollbar.set)
    
    def generate_report(self):
        """Generate a report based on selected options"""
        report_type = self.report_type.get()
        patient_id = self.patient_id.get() or "Unknown"
        patient_name = self.patient_name.get() or "Unknown"
        
        # Clear previous preview
        self.preview_text.delete(1.0, tk.END)
        
        # Report header
        header = f"MEDICAL DIAGNOSIS REPORT\n"
        header += f"{'='*50}\n\n"
        header += f"Report Type: {report_type}\n"
        header += f"Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        header += f"Patient ID: {patient_id}\n"
        header += f"Patient Name: {patient_name}\n\n"
        header += f"{'-'*50}\n\n"
        
        self.preview_text.insert(tk.END, header)
        
        # Report content based on type
        if report_type == "Current Diagnosis":
            self._generate_current_diagnosis_report()
        elif report_type == "History Summary":
            self._generate_history_summary_report()
        elif report_type == "Full Patient History":
            self._generate_full_history_report()
    
    def _generate_current_diagnosis_report(self):
        """Generate report for current diagnosis"""
        # Get latest diagnosis if available
        if not self.predictor.history:
            self.preview_text.insert(tk.END, "No diagnosis data available.\n")
            return
        
        latest = self.predictor.history[-1]
        symptoms = latest['symptoms']
        predictions = latest['predictions']
        
        # Add symptoms section
        self.preview_text.insert(tk.END, "PATIENT SYMPTOMS:\n")
        for symptom in symptoms:
            formatted_symptom = symptom.replace('_', ' ').title()
            self.preview_text.insert(tk.END, f"- {formatted_symptom}\n")
        
        self.preview_text.insert(tk.END, f"\n{'-'*50}\n\n")
        
        # Add diagnosis section
        self.preview_text.insert(tk.END, "DIAGNOSIS RESULTS:\n\n")
        
        for model_name, result in predictions.items():
            disease = result['disease']
            confidence = result['confidence']
            
            self.preview_text.insert(tk.END, f"Model: {model_name}\n")
            self.preview_text.insert(tk.END, f"Diagnosis: {disease}\n")
            self.preview_text.insert(tk.END, f"Confidence: {confidence:.1%}\n")
            self.preview_text.insert(tk.END, "\n")
        
        # Add recommendation section
        self.preview_text.insert(tk.END, f"{'-'*50}\n\n")
        self.preview_text.insert(tk.END, "RECOMMENDATIONS:\n\n")
        self.preview_text.insert(tk.END, "1. Please consult with a healthcare professional for a complete diagnosis.\n")
        self.preview_text.insert(tk.END, "2. This report is generated by an automated system and should be used as a reference only.\n")
        self.preview_text.insert(tk.END, "3. Further tests may be required to confirm the diagnosis.\n\n")
        
        # Footer
        self.preview_text.insert(tk.END, f"{'-'*50}\n\n")
        self.preview_text.insert(tk.END, "Report generated by Advanced Medical Diagnosis System\n")
    
    def _generate_history_summary_report(self):
        """Generate a summary of patient diagnosis history"""
        if not self.predictor.history:
            self.preview_text.insert(tk.END, "No diagnosis history available.\n")
            return
        
        # Add summary section
        self.preview_text.insert(tk.END, "DIAGNOSIS HISTORY SUMMARY:\n\n")
        self.preview_text.insert(tk.END, f"Total diagnoses: {len(self.predictor.history)}\n")
        self.preview_text.insert(tk.END, f"First diagnosis: {self.predictor.history[0]['timestamp']}\n")
        self.preview_text.insert(tk.END, f"Latest diagnosis: {self.predictor.history[-1]['timestamp']}\n\n")
        
        # Count disease occurrences
        disease_counts = {}
        for entry in self.predictor.history:
            for model_name, result in entry['predictions'].items():
                disease = result['disease']
                if disease not in disease_counts:
                    disease_counts[disease] = 0
                disease_counts[disease] += 1
        
        # Display most common diseases
        self.preview_text.insert(tk.END, "MOST COMMON DIAGNOSES:\n\n")
        sorted_diseases = sorted(disease_counts.items(), key=lambda x: x[1], reverse=True)
        
        for disease, count in sorted_diseases[:5]:  # Top 5 diseases
            percentage = count / len(self.predictor.history) * 100
            self.preview_text.insert(tk.END, f"{disease}: {count} occurrences ({percentage:.1f}%)\n")
    
    def _generate_full_history_report(self):
        """Generate full patient diagnosis history"""
        if not self.predictor.history:
            self.preview_text.insert(tk.END, "No diagnosis history available.\n")
            return
        
        self.preview_text.insert(tk.END, "FULL DIAGNOSIS HISTORY:\n\n")
        
        for i, entry in enumerate(self.predictor.history, 1):
            timestamp = entry['timestamp']
            symptoms = entry['symptoms']
            
            # Get consensus diagnosis (highest confidence)
            consensus_disease = None
            max_confidence = 0
            
            for model_name, result in entry['predictions'].items():
                if result['confidence'] > max_confidence:
                    max_confidence = result['confidence']
                    consensus_disease = result['disease']
            
            self.preview_text.insert(tk.END, f"Diagnosis #{i}: {timestamp}\n")
            self.preview_text.insert(tk.END, f"Symptoms: {', '.join(symptom.replace('_', ' ').title() for symptom in symptoms)}\n")
            self.preview_text.insert(tk.END, f"Diagnosis: {consensus_disease} (Confidence: {max_confidence:.1%})\n")
            self.preview_text.insert(tk.END, f"{'-'*30}\n\n")
    
    def export_as_pdf(self):
        """Export the current report as PDF"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
            title="Save Report as PDF"
        )
        
        if not file_path:
            return
        
        try:
            with PdfPages(file_path) as pdf:
                # Create a figure for text content
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.axis('off')
                
                # Get text content from preview
                report_text = self.preview_text.get(1.0, tk.END)
                
                # Add text to figure
                ax.text(0.1, 0.95, report_text, 
                        transform=ax.transAxes,
                        fontsize=10, 
                        verticalalignment='top',
                        family='monospace')
                
                # Save figure to PDF
                pdf.savefig(fig)
                plt.close(fig)
                
                # If visualization is included and we have current diagnosis
                if self.include_viz.get() and self.report_type.get() == "Current Diagnosis" and self.predictor.history:
                    # Create visualization similar to the one in the Visualization tab
                    fig, ax = plt.subplots(figsize=(8.5, 8))
                    
                    # Get the first model's predictions
                    latest = self.predictor.history[-1]
                    model_name = list(latest['predictions'].keys())[0]
                    probabilities = latest['predictions'][model_name]['probabilities']
                    
                    # Create bar plot
                    diseases = list(probabilities.keys())
                    values = list(probabilities.values())
                    
                    ax.bar(diseases, values)
                    ax.set_xticklabels(diseases, rotation=45, ha='right')
                    ax.set_ylabel('Probability')
                    ax.set_title(f'Disease Probabilities ({model_name})')
                    
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
            
            # Show success message
            tk.messagebox.showinfo("Export Successful", f"Report successfully exported to {os.path.basename(file_path)}")
            
        except Exception as e:
            tk.messagebox.showerror("Export Error", f"Error exporting PDF: {str(e)}")
    
    def export_as_csv(self):
        """Export the diagnosis history as CSV"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save History as CSV"
        )
        
        if not file_path:
            return
        
        try:
            # Create dataframe from history
            data = []
            
            for entry in self.predictor.history:
                timestamp = entry['timestamp']
                symptoms = ', '.join(entry['symptoms'])
                
                # Get consensus diagnosis (highest confidence)
                consensus_disease = None
                max_confidence = 0
                
                for model_name, result in entry['predictions'].items():
                    if result['confidence'] > max_confidence:
                        max_confidence = result['confidence']
                        consensus_disease = result['disease']
                
                data.append({
                    'Timestamp': timestamp,
                    'Symptoms': symptoms,
                    'Diagnosis': consensus_disease,
                    'Confidence': f"{max_confidence:.1%}"
                })
            
            # Create DataFrame and export
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            
            # Show success message
            tk.messagebox.showinfo("Export Successful", f"History successfully exported to {os.path.basename(file_path)}")
            
        except Exception as e:
            tk.messagebox.showerror("Export Error", f"Error exporting CSV: {str(e)}")

class ModernUI:
    def __init__(self, predictor):
        self.predictor = predictor
        self.root = tk.Tk()
        self.root.title("Advanced Medical Diagnosis System")
        self.root.geometry("1200x800")
        self.setup_ui()
        
        # Initialize report generator after UI setup
        self.report_generator = ReportGenerator(predictor, self)
        
    def setup_ui(self):
        # Configure styles
        self.setup_styles()
        
        # Create main container
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Create tabs
        self.create_tabs()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure(
            'Header.TLabel',
            font=('Helvetica', 24, 'bold'),
            padding=10
        )
        style.configure(
            'Tab.TNotebook',
            background='#ffffff',
            padding=5
        )
        
    def create_tabs(self):
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        
        # Create different tabs
        self.diagnosis_tab = self.create_diagnosis_tab()
        self.visualization_tab = self.create_visualization_tab()
        self.history_tab = self.create_history_tab()
        
        self.notebook.add(self.diagnosis_tab, text="Diagnosis")
        self.notebook.add(self.visualization_tab, text="Visualization")
        self.notebook.add(self.history_tab, text="History")
        
    def create_diagnosis_tab(self):
        tab = ttk.Frame(self.notebook, padding=20)
        
        # Header
        ttk.Label(
            tab,
            text="Medical Diagnosis System",
            style='Header.TLabel'
        ).grid(row=0, column=0, columnspan=2, pady=20)
        
        # Symptom selection
        self.symptom_vars = []
        for i, symptom in enumerate(self.predictor.symptoms):
            var = tk.BooleanVar()
            ttk.Checkbutton(
                tab,
                text=symptom.replace('_', ' ').title(),
                variable=var
            ).grid(row=i//3 + 1, column=i%3, sticky='w', padx=5, pady=5)
            self.symptom_vars.append((symptom, var))
            
        # Predict button
        ttk.Button(
            tab,
            text="Diagnose",
            command=self.make_prediction
        ).grid(row=len(self.predictor.symptoms)//3 + 2, column=0, columnspan=3, pady=20)
        
        # Results area
        self.results_frame = ttk.LabelFrame(tab, text="Diagnosis Results", padding=10)
        self.results_frame.grid(row=len(self.predictor.symptoms)//3 + 3, column=0, columnspan=3, sticky="ew")
        
        return tab
        
    def create_visualization_tab(self):
        tab = ttk.Frame(self.notebook, padding=20)
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        return tab
        
    def create_history_tab(self):
        tab = ttk.Frame(self.notebook, padding=20)
        
        # Create treeview for history
        columns = ('Date', 'Symptoms', 'Prediction', 'Confidence')
        self.history_tree = ttk.Treeview(tab, columns=columns, show='headings')
        
        # Configure columns
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=150)
            
        self.history_tree.grid(row=0, column=0, sticky="nsew")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(tab, orient="vertical", command=self.history_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        return tab
        
    def make_prediction(self):
        # Get selected symptoms
        selected_symptoms = [
            symptom for symptom, var in self.symptom_vars
            if var.get()
        ]
        
        if not selected_symptoms:
            messagebox.showwarning(
                "No Symptoms",
                "Please select at least one symptom"
            )
            return
            
        # Get predictions
        predictions = self.predictor.predict(selected_symptoms)
        
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
            
        # Display results
        for i, (model_name, result) in enumerate(predictions.items()):
            ttk.Label(
                self.results_frame,
                text=f"{model_name}:"
            ).grid(row=i, column=0, padx=5, pady=5)
            
            ttk.Label(
                self.results_frame,
                text=f"{result['disease']} ({result['confidence']:.1%})"
            ).grid(row=i, column=1, padx=5, pady=5)
            
        # Update visualization
        self.update_visualization(predictions)
        
        # Update history
        self.update_history(selected_symptoms, predictions)
        
    def update_visualization(self, predictions):
        self.ax.clear()
        
        # Get the first model's predictions for visualization
        model_name = list(predictions.keys())[0]
        probabilities = predictions[model_name]['probabilities']
        
        # Create bar plot
        diseases = list(probabilities.keys())
        values = list(probabilities.values())
        
        self.ax.bar(diseases, values)
        self.ax.set_xticklabels(diseases, rotation=45, ha='right')
        self.ax.set_ylabel('Probability')
        self.ax.set_title(f'Disease Probabilities ({model_name})')
        
        plt.tight_layout()
        self.canvas.draw()
        
    def update_history(self, symptoms, predictions):
        # Get the consensus prediction
        consensus = max(
            predictions.items(),
            key=lambda x: x[1]['confidence']
        )[1]['disease']
        
        # Add to treeview
        self.history_tree.insert(
            '',
            'end',
            values=(
                datetime.now().strftime('%Y-%m-%d %H:%M'),
                ', '.join(symptoms),
                consensus,
                f"{predictions['Random Forest']['confidence']:.1%}"
            )
        )
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    predictor = AdvancedDiseasePredictionSystem()
    app = ModernUI(predictor)
    app.run()