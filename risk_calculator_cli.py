#!/usr/bin/env python3
"""
CLI Risk Calculator - 100% FREE
================================
"""
from risk_calculator import calculate_ascvd_risk

def get_input(prompt, typ, mn=None, mx=None, default=None):
    while True:
        try:
            if default is not None:
                val = input(f"{prompt} [default: {default}]: ").strip()
                if not val: return default
            else:
                val = input(f"{prompt}: ").strip()
            val = typ(val)
            if mn and val < mn: print(f"  Min: {mn}"); continue
            if mx and val > mx: print(f"  Max: {mx}"); continue
            return val
        except ValueError: print(f"  Invalid {typ.__name__}")
        except KeyboardInterrupt: print("\nExiting..."); exit(0)

def get_yes_no(prompt, default=False):
    while True:
        resp = input(f"{prompt} ({'Y/n' if default else 'y/N'}): ").strip().lower()
        if not resp: return default
        if resp in ['y','yes']: return True
        if resp in ['n','no']: return False
        print("  Enter y or n")

def main():
    print("\n" + "="*70)
    print("   ASCVD 10-YEAR RISK CALCULATOR (CLI - FREE)")
    print("="*70 + "\n")
    
    print("Demographics:")
    age = get_input("  Age (40-79)", float, 40, 79)
    sex = "Male" if input("  Sex (M/F): ").strip().upper().startswith('M') else "Female"
    print("  Race: 1=White, 2=Black, 3=Other")
    race = ["White","Black","Other"][int(input("  Enter 1-3: "))-1]
    
    print("\nCholesterol:")
    tc = get_input("  Total Cholesterol (mg/dL, 130-320)", float, 130, 320)
    hdl = get_input("  HDL (mg/dL, 20-100)", float, 20, 100)
    
    print("\nBlood Pressure:")
    sbp = get_input("  Systolic BP (mmHg, 90-200)", float, 90, 200)
    bp_meds = get_yes_no("  On BP medication?")
    
    print("\nRisk Factors:")
    diab = get_yes_no("  Diabetes?")
    smoke = get_yes_no("  Current smoker?")
    
    print("\nOptional (Enter to skip):")
    bmi = get_input("  BMI (15-50)", float, 15, 50, None)
    hba1c = get_input("  HbA1c (4-14)", float, 4, 14, None)
    crp = get_input("  hs-CRP (0-20)", float, 0, 20, None)
    
    print("\n⏳ Calculating...")
    result = calculate_ascvd_risk(age, sex, race, tc, hdl, sbp, bp_meds, diab, smoke, bmi, hba1c, crp)
    
    colors = {"Low":"\033[92m","Borderline":"\033[93m","Intermediate":"\033[33m","High":"\033[91m"}
    col = colors[result["risk_category"]]
    
    print("\n" + "="*70)
    print(f"{col}10-YEAR ASCVD RISK: {result['risk_percent']:.1f}% ({result['risk_category']})\033[0m")
    print("="*70)
    print(f"\n{result['clinical_interpretation']}")
    
    if result['risk_enhancers']:
        print(f"\n{col}⚠ Risk Enhancers:\033[0m")
        for e in result['risk_enhancers']: print(f"  • {e}")
    
    if get_yes_no("\n💾 Save report?", True):
        with open("risk_report.txt", "w") as f:
            f.write(f"ASCVD Risk: {result['risk_percent']:.1f}% ({result['risk_category']})\n")
            f.write(f"{result['clinical_interpretation']}\n")
        print("✓ Saved to risk_report.txt")
    
    if get_yes_no("\nCalculate for another patient?"):
        main()
    else:
        print("\n✓ Stay heart healthy! ❤\n")

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: print("\n\nExiting...\n")
