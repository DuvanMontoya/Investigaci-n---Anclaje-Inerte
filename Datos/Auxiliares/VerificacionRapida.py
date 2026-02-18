
import pandas as pd
import numpy as np
import glob
import os
from sklearn.linear_model import LinearRegression

def analyze_calibration(data_dir):
    pattern = os.path.join(data_dir, "csigma_inerte_D-*_Rmax*_step10_Rmin101.csv")
    files = glob.glob(pattern)
    
    results = []
    
    print(f"Found {len(files)} files.")
    
    for filepath in sorted(files):
        # Extract Delta from filename
        filename = os.path.basename(filepath)
        # format: csigma_inerte_D-3_...
        try:
            parts = filename.split('_')
            delta_str = parts[2] # D-3
            delta = int(delta_str.replace('D', ''))
            # Data assumes D is negative based on the context (Heegner), but filename says D-3.
            # Let's check the content 'D' column to be sure.
        except:
            delta = "Unknown"
            
        try:
            df = pd.read_csv(filepath)
            if 'D' in df.columns:
                delta = df['D'].iloc[0]
            
            # 1. Stabilization Analysis (Tail Regression)
            # Model: C(R) = C_inf + a * (1/log R)
            
            cutoffs = [1_000_000, 2_000_000]
            
            row_result = {'Delta': delta}
            
            for cut in cutoffs:
                mask = df['R'] >= cut
                df_tail = df[mask].copy()
                
                if len(df_tail) == 0:
                    row_result[f'C_inf_{cut}'] = np.nan
                    row_result[f'Slope_{cut}'] = np.nan
                    continue
                
                Y = df_tail['C_sigma_inerte'].values
                X = 1.0 / np.log(df_tail['R'].values)
                X = X.reshape(-1, 1)
                
                reg = LinearRegression().fit(X, Y)
                
                c_inf = reg.intercept_
                slope = reg.coef_[0]
                r2 = reg.score(X, Y)
                
                # Deviation statistics
                std_dev = np.std(Y)
                
                row_result[f'C_inf_{cut}'] = c_inf
                row_result[f'Slope_{cut}'] = slope
                row_result[f'R2_{cut}'] = r2
                row_result[f'Std_{cut}'] = std_dev
                row_result[f'N_points_{cut}'] = len(df_tail)
            
            # Collapse verification (comparing Raw vs Corrected if possible)
            # We don't have strictly "Raw" C unless we reverse the H_sigma correction.
            # But the user asks primarily for the table of C_inf.
            
            results.append(row_result)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    # Sort by absolute delta if possible
    summary_df['AbsDelta'] = summary_df['Delta'].abs()
    summary_df = summary_df.sort_values('AbsDelta').drop('AbsDelta', axis=1)
    
    # Check consistency between cutoffs
    summary_df['Sensitivity'] = summary_df['C_inf_1000000'] - summary_df['C_inf_2000000']
    
    return summary_df

if __name__ == "__main__":
    data_directory = "Datos/Canonicos"
    df_results = analyze_calibration(data_directory)
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.6f}'.format)
    
    print("\nanalysis_results_table:")
    print(df_results)
    
    # Calculate the average C_infinity (Renormalized Singular Series check)
    # The user mentioned C_base(Delta). 
    # If the user wants C_infinity / C_base = S_Delta (empirical), I might need C_base.
    # C_base is usually the standard Hardy-Littlewood constant.
    # For R^2 - c^2 it's related to the product over p of (1 - chi(D)/p)/(1-1/p)...
    # But since I don't have the formula for C_base implemented, I will report C_inf explicitly.
