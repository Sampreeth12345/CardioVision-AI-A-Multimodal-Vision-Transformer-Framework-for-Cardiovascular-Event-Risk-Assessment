"""
ASCVD 10-Year Risk Calculator - FREE VERSION
=============================================
100% FREE - No API keys required!
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class PatientData:
    age: float
    sex: str
    race: str
    total_cholesterol: float
    hdl_cholesterol: float
    systolic_bp: float
    on_bp_medication: bool
    diabetes: bool
    current_smoker: bool
    bmi: float = None
    hba1c: float = None
    hs_crp: float = None
    
    def validate(self) -> Tuple[bool, str]:
        if not (40 <= self.age <= 79):
            return False, "Age must be 40-79 years"
        if self.sex not in ["Male", "Female"]:
            return False, "Sex must be Male or Female"
        if self.race not in ["White", "Black", "Other"]:
            return False, "Race must be White, Black, or Other"
        if not (130 <= self.total_cholesterol <= 320):
            return False, "Total cholesterol must be 130-320 mg/dL"
        if not (20 <= self.hdl_cholesterol <= 100):
            return False, "HDL must be 20-100 mg/dL"
        if not (90 <= self.systolic_bp <= 200):
            return False, "Systolic BP must be 90-200 mmHg"
        return True, "Valid"

class ASCVDRiskCalculator:
    COEFFICIENTS = {
        "White_Male": {
            "ln_age": 12.344, "ln_age_squared": 0.0,
            "ln_total_chol": 11.853, "ln_age_ln_total_chol": -2.664,
            "ln_hdl": -7.990, "ln_age_ln_hdl": 1.769,
            "ln_treated_sbp": 1.797, "ln_untreated_sbp": 1.764,
            "current_smoker": 7.837, "ln_age_smoker": -1.795,
            "diabetes": 0.658, "baseline_survival": 0.9144,
            "mean_coefficient_value": 61.18
        },
        "Black_Male": {
            "ln_age": 2.469, "ln_age_squared": 0.0,
            "ln_total_chol": 0.302, "ln_age_ln_total_chol": 0.0,
            "ln_hdl": -0.307, "ln_age_ln_hdl": 0.0,
            "ln_treated_sbp": 1.916, "ln_untreated_sbp": 1.809,
            "current_smoker": 0.549, "ln_age_smoker": 0.0,
            "diabetes": 0.645, "baseline_survival": 0.8954,
            "mean_coefficient_value": 19.54
        },
        "White_Female": {
            "ln_age": -29.799, "ln_age_squared": 4.884,
            "ln_total_chol": 13.540, "ln_age_ln_total_chol": -3.114,
            "ln_hdl": -13.578, "ln_age_ln_hdl": 3.149,
            "ln_treated_sbp": 2.019, "ln_untreated_sbp": 1.957,
            "current_smoker": 7.574, "ln_age_smoker": -1.665,
            "diabetes": 0.661, "baseline_survival": 0.9665,
            "mean_coefficient_value": -29.18
        },
        "Black_Female": {
            "ln_age": 17.114, "ln_age_squared": 0.0,
            "ln_total_chol": 0.940, "ln_age_ln_total_chol": 0.0,
            "ln_hdl": -18.920, "ln_age_ln_hdl": 4.475,
            "ln_treated_sbp": 29.291, "ln_age_smoker": 0.0,
            "ln_untreated_sbp": 27.820, "current_smoker": 0.8738,
            "diabetes": 0.8738, "baseline_survival": 0.9533,
            "mean_coefficient_value": 203.100
        }
    }
    
    def calculate_risk(self, patient: PatientData) -> Dict:
        valid, message = patient.validate()
        if not valid:
            raise ValueError(f"Invalid: {message}")
        
        race_key = patient.race if patient.race in ["White", "Black"] else "White"
        coef_key = f"{race_key}_{patient.sex}"
        coef = self.COEFFICIENTS[coef_key]
        
        ln_age = np.log(patient.age)
        ln_age_squared = ln_age ** 2
        ln_total_chol = np.log(patient.total_cholesterol)
        ln_hdl = np.log(patient.hdl_cholesterol)
        
        if patient.on_bp_medication:
            ln_sbp_term = np.log(patient.systolic_bp) * coef["ln_treated_sbp"]
        else:
            ln_sbp_term = np.log(patient.systolic_bp) * coef["ln_untreated_sbp"]
        
        individual_sum = (
            coef["ln_age"] * ln_age +
            coef["ln_age_squared"] * ln_age_squared +
            coef["ln_total_chol"] * ln_total_chol +
            coef["ln_age_ln_total_chol"] * ln_age * ln_total_chol +
            coef["ln_hdl"] * ln_hdl +
            coef["ln_age_ln_hdl"] * ln_age * ln_hdl +
            ln_sbp_term +
            coef["current_smoker"] * (1 if patient.current_smoker else 0) +
            coef["ln_age_smoker"] * ln_age * (1 if patient.current_smoker else 0) +
            coef["diabetes"] * (1 if patient.diabetes else 0)
        )
        
        baseline_survival = coef["baseline_survival"]
        mean_sum = coef["mean_coefficient_value"]
        
        risk_decimal = 1 - (baseline_survival ** np.exp(individual_sum - mean_sum))
        risk_percent = risk_decimal * 100
        
        if risk_percent < 5:
            category = "Low"
        elif risk_percent < 7.5:
            category = "Borderline"
        elif risk_percent < 20:
            category = "Intermediate"
        else:
            category = "High"
        
        enhancers = []
        if patient.bmi and patient.bmi >= 30:
            enhancers.append("Obesity (BMI ≥30)")
        if patient.hba1c:
            if patient.hba1c >= 6.5:
                enhancers.append("Poor glycemic control (HbA1c ≥6.5%)")
            elif patient.hba1c >= 5.7:
                enhancers.append("Prediabetes (HbA1c 5.7-6.4%)")
        if patient.hs_crp and patient.hs_crp >= 2.0:
            enhancers.append("Elevated inflammation (hs-CRP ≥2.0 mg/L)")
        
        non_hdl = patient.total_cholesterol - patient.hdl_cholesterol
        if non_hdl >= 160:
            enhancers.append("Elevated non-HDL (≥160 mg/dL)")
        
        escalation_warranted = False
        if category in ["Borderline", "Intermediate"] and enhancers:
            escalation_warranted = True
        
        interp = f"Patient has {risk_percent:.1f}% 10-year ASCVD risk ({category} risk). "
        if category == "Low":
            interp += "Focus on healthy lifestyle. Statin generally not indicated."
        elif category == "Borderline":
            if enhancers:
                interp += f"Risk enhancers present. Consider moderate-intensity statin and CAC scoring."
            else:
                interp += "Risk discussion recommended."
        elif category == "Intermediate":
            interp += "Moderate-intensity statin recommended. Consider CAC scoring if uncertain."
        else:
            interp += "High-intensity statin strongly recommended. Consider cardiology referral."
        
        return {
            "risk_percent": round(risk_percent, 2),
            "risk_category": category,
            "risk_enhancers": enhancers,
            "escalation_warranted": escalation_warranted,
            "clinical_interpretation": interp
        }

def calculate_ascvd_risk(age, sex, race, total_cholesterol, hdl_cholesterol,
                         systolic_bp, on_bp_medication, diabetes, current_smoker,
                         bmi=None, hba1c=None, hs_crp=None) -> Dict:
    patient = PatientData(age, sex, race, total_cholesterol, hdl_cholesterol,
                          systolic_bp, on_bp_medication, diabetes, current_smoker,
                          bmi, hba1c, hs_crp)
    calculator = ASCVDRiskCalculator()
    return calculator.calculate_risk(patient)

def get_risk_category_color(category: str) -> str:
    colors = {"Low": "#2ecc71", "Borderline": "#f39c12",
              "Intermediate": "#e67e22", "High": "#e74c3c"}
    return colors.get(category, "#95a5a6")
