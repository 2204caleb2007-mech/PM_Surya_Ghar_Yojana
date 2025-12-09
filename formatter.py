import json
from datetime import datetime
import pandas as pd
from io import StringIO

class Formatter:
    @staticmethod
    def format_json(data, indent=2):
        """Format data as JSON"""
        return json.dumps(data, indent=indent, default=str)
    
    @staticmethod
    def format_csv(data_list):
        """Format list of dictionaries as CSV"""
        if not data_list:
            return ""
        
        df = pd.DataFrame(data_list)
        return df.to_csv(index=False)
    
    @staticmethod
    def format_summary(analysis_results):
        """Create human-readable summary"""
        summary = f"""
        Solar Analysis Summary
        =====================
        
        Location: {analysis_results.get('lat', 'N/A')}, {analysis_results.get('lon', 'N/A')}
        Analysis Date: {analysis_results.get('analysis_timestamp', 'N/A')}
        
        Detection Results:
        - Solar Panels Detected: {'Yes' if analysis_results.get('has_solar') else 'No'}
        - Confidence Score: {analysis_results.get('confidence', 0) * 100:.1f}%
        - Estimated Panel Count: {analysis_results.get('panel_count_est', 0)}
        - Estimated Area: {analysis_results.get('pv_area_sqm_est', 0)} m²
        - Estimated Capacity: {analysis_results.get('capacity_kw_est', 0)} kW
        
        Quality Control:
        - QC Status: {analysis_results.get('qc_status', 'N/A')}
        - QC Notes: {', '.join(analysis_results.get('qc_notes', []))}
        
        Power Generation:
        - Annual Generation: {analysis_results.get('annual_generation_kwh', 0)} kWh
        - Soiling Level: {analysis_results.get('soiling_level', 'N/A')}
        - Efficiency Loss: {analysis_results.get('efficiency_loss_percent', 0)}%
        
        Financial Analysis:
        - Estimated Cost: ₹{analysis_results.get('financial_analysis', {}).get('estimated_cost', 0):,}
        - Subsidy Available: ₹{analysis_results.get('financial_analysis', {}).get('subsidy_available', 0):,}
        - Payback Period: {analysis_results.get('financial_analysis', {}).get('payback_period_years', 0)} years
        """
        
        return summary
    
    @staticmethod
    def generate_report_id(latitude, longitude):
        """Generate unique report ID"""
        lat_str = f"{abs(latitude):.4f}{'N' if latitude >= 0 else 'S'}"
        lon_str = f"{abs(longitude):.4f}{'E' if longitude >= 0 else 'W'}"
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        
        return f"SOLAR_{lat_str}_{lon_str}_{timestamp}"