"""
LLM Integration - 100% FREE VERSION
====================================
No API keys, no internet, no cost!
Uses built-in medical guideline templates.
"""

import numpy as np
from typing import Dict, Optional

LABEL_MAPPING = {0: "Normal", 1: "Mild", 2: "Severe"}

GUIDELINES = {
    "Normal": {
        "treatment": "Focus on lifestyle: Mediterranean/DASH diet, 150min/week exercise, BMI 18.5-24.9, BP <120/80, LDL <100. Statin NOT recommended unless LDL ≥190. Follow-up every 4-6 years. [2019 ACC/AHA Sec 4.1]",
        "lifestyle": "Diet: vegetables, fruits, whole grains, fish, olive oil, nuts. Limit: sodium <2300mg/day, saturated fat, added sugars. Exercise: 150min moderate or 75min vigorous weekly. No tobacco. Weight: BMI 18.5-24.9. [2019 ACC/AHA Sec 3.2]",
        "followup": "Routine assessment every 4-6 years. Lipid panel every 4-6 years. BP at every visit. Annual monitoring if borderline risk factors emerge. [2019 ACC/AHA Sec 8.1]"
    },
    "Mild": {
        "treatment": "Risk discussion essential. Consider moderate statin if enhancers present (family history, LDL ≥160, metabolic syndrome, CKD, inflammation). Consider CAC scoring if uncertain. CAC >100 supports statin. [2019 ACC/AHA Sec 4.2]",
        "medication": "Moderate-intensity statin (30-49% LDL reduction): Atorvastatin 10-20mg or Rosuvastatin 5-10mg daily. Goal: ≥30% LDL reduction. Monitor at 4-12 weeks, then every 3-6 months. Screen for muscle symptoms. [2019 ACC/AHA Sec 5.3]",
        "testing": "Consider: CAC score (Class IIa), hs-CRP, lipoprotein(a), apolipoprotein B, ankle-brachial index. CAC=0: defer statin. CAC 1-99: consider if ≥75th percentile. CAC ≥100: recommend statin. [2019 ACC/AHA Sec 6.2]",
        "followup": "Initially every 3-6 months until stable, then 6-12 months. BP every 3-6 months. Annual comprehensive reassessment. Check lipids 4-12 weeks after statin start. [2019 ACC/AHA Sec 8.2]"
    },
    "Severe": {
        "treatment": "High-intensity statin STRONGLY recommended. Goal: LDL reduction ≥50%, target <70mg/dL. If LDL ≥70 despite max statin, add ezetimibe. Aspirin 75-100mg if low bleeding risk. BP target <130/80. [2019 ACC/AHA Sec 4.3]",
        "medication": "High-intensity statin (≥50% LDL reduction): Atorvastatin 40-80mg or Rosuvastatin 20-40mg daily. Add ezetimibe 10mg if LDL ≥70 on max statin. [2019 ACC/AHA Sec 5.4]",
        "lifestyle": "Sodium <1500mg/day. Saturated fat <6% calories. NO trans fat. Smoking cessation MANDATORY. Exercise ≥150min/week. Weight loss 5-10% if BMI ≥25. Stress management. [2019 ACC/AHA Sec 7.3]",
        "testing": "12-lead ECG. Lipid panel every 3 months until goal, then every 6 months. HbA1c every 3-6 months if diabetic. Renal function every 6-12 months. Consider echo if heart failure symptoms. [2019 ACC/AHA Sec 6.3]",
        "followup": "Lipids every 3 months until LDL goal, then every 6 months. BP EVERY visit. Muscle symptoms screening every visit. CARDIOLOGY REFERRAL recommended. Cardiac rehab enrollment. [2019 ACC/AHA Sec 8.3]"
    }
}

def get_recommendations(prediction: int, confidence_scores: np.ndarray,
                       clinical_data: Optional[Dict] = None) -> Dict[str, str]:
    """
    Generate recommendations - 100% FREE!
    No API keys, no internet required.
    """
    severity = LABEL_MAPPING[prediction]
    guidelines = GUIDELINES[severity]
    
    diagnosis = f"""**Classification**: {severity} Stenosis Risk

**Model Confidence**:
- Normal: {confidence_scores[0]*100:.1f}%
- Mild: {confidence_scores[1]*100:.1f}%  
- Severe: {confidence_scores[2]*100:.1f}%

The fusion model predicts {severity.lower()} coronary stenosis with {confidence_scores[prediction]*100:.1f}% confidence."""
    
    if severity == "Normal":
        diagnosis += "\n\nNo significant stenosis. Focus on primary prevention."
    elif severity == "Mild":
        diagnosis += "\n\nMild stenosis (20-60% narrowing). Moderate cardiovascular risk."
    else:
        diagnosis += "\n\nSevere stenosis (≥60% narrowing). High risk requiring aggressive management."
    
    return {
        "diagnosis_explanation": diagnosis,
        "lifestyle_modifications": guidelines.get("lifestyle", guidelines.get("treatment", "See treatment section")),
        "medication_guidance": guidelines.get("medication", guidelines.get("treatment", "Consult cardiologist")),
        "testing_recommendations": guidelines.get("testing", "Standard cardiovascular workup"),
        "followup_schedule": guidelines.get("followup", "Follow up with primary care")
    }
