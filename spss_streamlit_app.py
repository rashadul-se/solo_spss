"""
SPSS-Style Statistical Analysis Streamlit App
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
import io
import sys

# Set page configuration
st.set_page_config(
    page_title="SPSS-Style Statistical Analysis",
    page_icon="📊",
    layout="wide"
)

class SPSSAnalyzer:
    def __init__(self):
        self.data = None
        self.output_buffer = io.StringIO()
        
    def print_to_buffer(self, text):
        """Print to string buffer instead of console"""
        self.output_buffer.write(text + "\n")
    
    def get_output(self):
        """Get accumulated output"""
        return self.output_buffer.getvalue()
    
    def clear_output(self):
        """Clear output buffer"""
        self.output_buffer = io.StringIO()
    
    def print_header(self, title):
        """Print SPSS-style section header"""
        self.print_to_buffer("\n" + "="*70)
        self.print_to_buffer(f" {title}")
        self.print_to_buffer("="*70 + "\n")
    
    def print_table_header(self, columns, widths=None):
        """Print SPSS-style table header"""
        if widths is None:
            widths = [20] * len(columns)
        
        header = ""
        separator = ""
        for col, width in zip(columns, widths):
            header += f"{col:<{width}}"
            separator += "-" * width
        
        self.print_to_buffer(header)
        self.print_to_buffer(separator)
    
    def frequencies(self, variable):
        """Frequencies analysis"""
        self.print_header(f"FREQUENCIES: {variable}")
        
        self.print_to_buffer(f"FREQUENCIES VARIABLES={variable}")
        self.print_to_buffer("  /ORDER=ANALYSIS.")
        
        freq_table = self.data[variable].value_counts().sort_index()
        total = len(self.data[variable].dropna())
        
        self.print_to_buffer(f"\n{variable}")
        self.print_to_buffer("-" * 70)
        self.print_table_header(["Value", "Frequency", "Percent", "Valid Percent", "Cumulative"], 
                               [15, 12, 12, 15, 16])
        
        cumulative = 0
        for value, count in freq_table.items():
            percent = (count / len(self.data)) * 100
            valid_percent = (count / total) * 100
            cumulative += valid_percent
            
            self.print_to_buffer(f"{str(value):<15}{count:<12}{percent:<12.1f}{valid_percent:<15.1f}{cumulative:<16.1f}")
        
        n_missing = self.data[variable].isna().sum()
        if n_missing > 0:
            missing_pct = (n_missing / len(self.data)) * 100
            self.print_to_buffer(f"{'Missing':<15}{n_missing:<12}{missing_pct:<12.1f}")
        
        self.print_to_buffer(f"{'Total':<15}{len(self.data):<12}{'100.0':<12}")
    
    def descriptives(self, variables):
        """Descriptive statistics"""
        self.print_header("DESCRIPTIVE STATISTICS")
        
        self.print_to_buffer(f"DESCRIPTIVES VARIABLES={' '.join(variables)}")
        self.print_to_buffer("  /STATISTICS=MEAN STDDEV MIN MAX.")
        
        self.print_to_buffer("\nDescriptive Statistics")
        self.print_to_buffer("-" * 90)
        self.print_table_header(["Variable", "N", "Minimum", "Maximum", "Mean", "Std. Deviation"], 
                               [20, 10, 12, 12, 12, 14])
        
        for var in variables:
            data_clean = self.data[var].dropna()
            n = len(data_clean)
            minimum = data_clean.min()
            maximum = data_clean.max()
            mean = data_clean.mean()
            std = data_clean.std()
            
            self.print_to_buffer(f"{var:<20}{n:<10}{minimum:<12.2f}{maximum:<12.2f}{mean:<12.2f}{std:<14.2f}")
    
    def crosstabs(self, row_var, col_var, chi_square=True):
        """Crosstabs analysis"""
        self.print_header(f"CROSSTABS: {row_var} * {col_var}")
        
        self.print_to_buffer(f"CROSSTABS")
        self.print_to_buffer(f"  /TABLES={row_var} BY {col_var}")
        self.print_to_buffer(f"  /FORMAT=AVALUE TABLES")
        self.print_to_buffer(f"  /STATISTICS=CHISQ")
        self.print_to_buffer(f"  /CELLS=COUNT ROW COLUMN TOTAL.")
        
        crosstab = pd.crosstab(self.data[row_var], self.data[col_var], margins=True)
        
        self.print_to_buffer(f"\n{row_var} * {col_var} Crosstabulation")
        self.print_to_buffer("-" * 70)
        self.print_to_buffer(str(crosstab))
        
        if chi_square:
            self.print_to_buffer("\n")
            self.print_header("Chi-Square Tests")
            
            ct = pd.crosstab(self.data[row_var], self.data[col_var])
            chi2, p_value, dof, expected = chi2_contingency(ct)
            
            self.print_to_buffer("Chi-Square Tests")
            self.print_to_buffer("-" * 70)
            self.print_table_header(["Test", "Value", "df", "Asymp. Sig. (2-sided)"], 
                                   [30, 15, 10, 25])
            
            self.print_to_buffer(f"{'Pearson Chi-Square':<30}{chi2:<15.3f}{dof:<10}{p_value:<25.3f}")
            self.print_to_buffer(f"{'N of Valid Cases':<30}{len(self.data):<15}")
            
            if p_value < 0.05:
                self.print_to_buffer(f"\na. {p_value:.3f} < 0.05: Significant association found.")
            else:
                self.print_to_buffer(f"\na. {p_value:.3f} > 0.05: No significant association found.")
    
    def independent_t_test(self, dependent_var, grouping_var):
        """Independent Samples T-Test"""
        self.print_header("INDEPENDENT SAMPLES T-TEST")
        
        self.print_to_buffer(f"T-TEST GROUPS={grouping_var}(1 2)")
        self.print_to_buffer(f"  /MISSING=ANALYSIS")
        self.print_to_buffer(f"  /VARIABLES={dependent_var}")
        self.print_to_buffer(f"  /CRITERIA=CI(.95).")
        
        groups = self.data[grouping_var].unique()
        group1_data = self.data[self.data[grouping_var] == groups[0]][dependent_var].dropna()
        group2_data = self.data[self.data[grouping_var] == groups[1]][dependent_var].dropna()
        
        # Group Statistics
        self.print_to_buffer("\nGroup Statistics")
        self.print_to_buffer("-" * 80)
        self.print_table_header([grouping_var, "N", "Mean", "Std. Deviation", "Std. Error Mean"], 
                               [20, 10, 15, 18, 17])
        
        self.print_to_buffer(f"{str(groups[0]):<20}{len(group1_data):<10}{group1_data.mean():<15.2f}"
              f"{group1_data.std():<18.2f}{group1_data.sem():<17.2f}")
        self.print_to_buffer(f"{str(groups[1]):<20}{len(group2_data):<10}{group2_data.mean():<15.2f}"
              f"{group2_data.std():<18.2f}{group2_data.sem():<17.2f}")
        
        # T-Test
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
        
        self.print_to_buffer("\n\nIndependent Samples Test")
        self.print_to_buffer("-" * 80)
        self.print_table_header(["", "t", "df", "Sig. (2-tailed)", "Mean Difference"], 
                               [25, 12, 10, 18, 15])
        
        df = len(group1_data) + len(group2_data) - 2
        mean_diff = group1_data.mean() - group2_data.mean()
        
        self.print_to_buffer(f"{dependent_var:<25}{t_stat:<12.3f}{df:<10}{p_value:<18.3f}{mean_diff:<15.2f}")
        
        if p_value < 0.05:
            self.print_to_buffer(f"\nSignificant difference found (p = {p_value:.3f} < 0.05)")
        else:
            self.print_to_buffer(f"\nNo significant difference found (p = {p_value:.3f} > 0.05)")
    
    def paired_t_test(self, var1, var2):
        """Paired Samples T-Test"""
        self.print_header("PAIRED SAMPLES T-TEST")
        
        self.print_to_buffer(f"T-TEST PAIRS={var1} WITH {var2} (PAIRED)")
        self.print_to_buffer(f"  /CRITERIA=CI(.95)")
        self.print_to_buffer(f"  /MISSING=ANALYSIS.")
        
        data1 = self.data[var1].dropna()
        data2 = self.data[var2].dropna()
        
        # Paired Statistics
        self.print_to_buffer("\nPaired Samples Statistics")
        self.print_to_buffer("-" * 70)
        self.print_table_header(["Variable", "Mean", "N", "Std. Deviation", "Std. Error Mean"], 
                               [15, 12, 10, 18, 17])
        
        self.print_to_buffer(f"{var1:<15}{data1.mean():<12.2f}{len(data1):<10}{data1.std():<18.2f}{data1.sem():<17.2f}")
        self.print_to_buffer(f"{var2:<15}{data2.mean():<12.2f}{len(data2):<10}{data2.std():<18.2f}{data2.sem():<17.2f}")
        
        # T-Test
        t_stat, p_value = stats.ttest_rel(data1, data2)
        
        self.print_to_buffer("\n\nPaired Samples Test")
        self.print_to_buffer("-" * 70)
        self.print_table_header(["Pair", "t", "df", "Sig. (2-tailed)"], 
                               [30, 12, 10, 18])
        
        df = len(data1) - 1
        
        self.print_to_buffer(f"{f'{var1} - {var2}':<30}{t_stat:<12.3f}{df:<10}{p_value:<18.3f}")
        
        if p_value < 0.05:
            self.print_to_buffer(f"\nSignificant difference found (p = {p_value:.3f} < 0.05)")
        else:
            self.print_to_buffer(f"\nNo significant difference found (p = {p_value:.3f} > 0.05)")
    
    def one_way_anova(self, dependent_var, factor_var):
        """One-Way ANOVA"""
        self.print_header("ONE-WAY ANOVA")
        
        self.print_to_buffer(f"ONEWAY {dependent_var} BY {factor_var}")
        self.print_to_buffer(f"  /STATISTICS DESCRIPTIVES HOMOGENEITY")
        self.print_to_buffer(f"  /MISSING ANALYSIS.")
        
        # Descriptives
        self.print_to_buffer("\nDescriptives")
        self.print_to_buffer("-" * 80)
        self.print_table_header(["Group", "N", "Mean", "Std. Deviation", "Std. Error"], 
                               [15, 10, 15, 18, 12])
        
        groups = self.data[factor_var].unique()
        group_data = []
        
        for group in groups:
            data = self.data[self.data[factor_var] == group][dependent_var].dropna()
            group_data.append(data)
            self.print_to_buffer(f"{str(group):<15}{len(data):<10}{data.mean():<15.2f}"
                  f"{data.std():<18.2f}{data.sem():<12.2f}")
        
        # ANOVA Table
        f_stat, p_value = stats.f_oneway(*group_data)
        
        total_n = sum(len(g) for g in group_data)
        between_df = len(groups) - 1
        within_df = total_n - len(groups)
        total_df = total_n - 1
        
        self.print_to_buffer("\n\nANOVA")
        self.print_to_buffer("-" * 70)
        self.print_table_header(["Source", "Sum of Squares", "df", "Mean Square", "F", "Sig."], 
                               [15, 18, 8, 15, 12, 12])
        
        self.print_to_buffer(f"{'Between Groups':<15}{'':18}{between_df:<8}{'':15}{f_stat:<12.3f}{p_value:<12.3f}")
        self.print_to_buffer(f"{'Within Groups':<15}{'':18}{within_df:<8}")
        self.print_to_buffer(f"{'Total':<15}{'':18}{total_df:<8}")
        
        if p_value < 0.05:
            self.print_to_buffer(f"\nSignificant difference found (p = {p_value:.3f} < 0.05)")
        else:
            self.print_to_buffer(f"\nNo significant difference found (p = {p_value:.3f} > 0.05)")
    
    def correlation(self, var1, var2, method='pearson'):
        """Correlation analysis"""
        self.print_header("CORRELATIONS")
        
        self.print_to_buffer(f"CORRELATIONS")
        self.print_to_buffer(f"  /VARIABLES={var1} {var2}")
        self.print_to_buffer(f"  /PRINT={method.upper()} TWOTAIL NOSIG")
        self.print_to_buffer(f"  /MISSING=PAIRWISE.")
        
        data1 = self.data[var1].dropna()
        data2 = self.data[var2].dropna()
        
        if method.lower() == 'pearson':
            corr, p_value = pearsonr(data1, data2)
            method_name = "Pearson Correlation"
        else:
            corr, p_value = spearmanr(data1, data2)
            method_name = "Spearman's rho"
        
        self.print_to_buffer(f"\n{method_name}")
        self.print_to_buffer("-" * 70)
        self.print_to_buffer(f"\n{'':20}{var1:>15}{var2:>15}")
        self.print_to_buffer(f"{var1:<20}{'Correlation':>15}{'1.000':>15}{corr:>15.3f}")
        self.print_to_buffer(f"{'':20}{'Sig. (2-tailed)':>15}{' ':>15}{p_value:>15.3f}")
        self.print_to_buffer(f"{'':20}{'N':>15}{len(data1):>15}{len(data1):>15}")
        
        self.print_to_buffer(f"\n{var2:<20}{'Correlation':>15}{corr:>15.3f}{'1.000':>15}")
        self.print_to_buffer(f"{'':20}{'Sig. (2-tailed)':>15}{p_value:>15}{' ':>15}")
        self.print_to_buffer(f"{'':20}{'N':>15}{len(data2):>15}{len(data2):>15}")
        
        if p_value < 0.05:
            self.print_to_buffer(f"\n**. Correlation is significant at the 0.05 level (2-tailed).")


# Generate dummy data
def generate_dummy_data():
    np.random.seed(42)
    n = 200
    
    data = pd.DataFrame({
        'Student_ID': range(1, n + 1),
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Age': np.random.randint(18, 25, n),
        'Education_Level': np.random.choice(['Freshman', 'Sophomore', 'Junior', 'Senior'], n),
        'Study_Hours': np.random.normal(15, 5, n).clip(0, 40),
        'PreTest_Score': np.random.normal(65, 10, n).clip(0, 100),
        'PostTest_Score': np.random.normal(75, 10, n).clip(0, 100),
        'Final_Grade': np.random.normal(75, 12, n).clip(0, 100),
        'Attendance': np.random.normal(85, 10, n).clip(0, 100),
        'Satisfaction': np.random.choice(['Very Unsatisfied', 'Unsatisfied', 'Neutral', 'Satisfied', 'Very Satisfied'], n)
    })
    
    return data


# Streamlit UI
def main():
    st.title("📊 SPSS-Style Statistical Analysis Tool")
    st.markdown("### For Undergraduate Thesis & Capstone Projects")
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SPSSAnalyzer()
    
    if 'data' not in st.session_state:
        st.session_state.data = generate_dummy_data()
        st.session_state.analyzer.data = st.session_state.data
    
    # Sidebar
    st.sidebar.header("📁 Data Management")
    
    if st.sidebar.button("🔄 Generate New Dummy Data"):
        st.session_state.data = generate_dummy_data()
        st.session_state.analyzer.data = st.session_state.data
        st.success("New dummy data generated!")
    
    uploaded_file = st.sidebar.file_uploader("📂 Upload Your Data (CSV/Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.data = pd.read_csv(uploaded_file)
            else:
                st.session_state.data = pd.read_excel(uploaded_file)
            
            st.session_state.analyzer.data = st.session_state.data
            st.sidebar.success("✅ Data uploaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
    
    # Display data
    st.header("📋 Data View")
    st.dataframe(st.session_state.data.head(10), use_container_width=True)
    st.caption(f"Total rows: {len(st.session_state.data)} | Total columns: {len(st.session_state.data.columns)}")
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Descriptive Statistics", "🧪 Inferential Statistics", "📈 Correlation & Regression", "🔍 Advanced"])
    
    # Tab 1: Descriptive Statistics
    with tab1:
        st.header("Descriptive Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Frequencies")
            freq_var = st.selectbox("Select Variable for Frequencies:", 
                                   st.session_state.data.columns,
                                   key='freq_var')
            
            if st.button("Run Frequencies"):
                st.session_state.analyzer.clear_output()
                st.session_state.analyzer.frequencies(freq_var)
                output = st.session_state.analyzer.get_output()
                st.code(output, language='text')
        
        with col2:
            st.subheader("Descriptives")
            desc_vars = st.multiselect("Select Variables for Descriptives:", 
                                      st.session_state.data.select_dtypes(include=[np.number]).columns,
                                      key='desc_vars')
            
            if st.button("Run Descriptives") and desc_vars:
                st.session_state.analyzer.clear_output()
                st.session_state.analyzer.descriptives(desc_vars)
                output = st.session_state.analyzer.get_output()
                st.code(output, language='text')
        
        st.subheader("Crosstabs")
        col3, col4 = st.columns(2)
        
        with col3:
            row_var = st.selectbox("Row Variable:", st.session_state.data.columns, key='row_var')
        
        with col4:
            col_var = st.selectbox("Column Variable:", st.session_state.data.columns, key='col_var')
        
        chi_square_check = st.checkbox("Include Chi-Square Test", value=True)
        
        if st.button("Run Crosstabs"):
            st.session_state.analyzer.clear_output()
            st.session_state.analyzer.crosstabs(row_var, col_var, chi_square=chi_square_check)
            output = st.session_state.analyzer.get_output()
            st.code(output, language='text')
    
    # Tab 2: Inferential Statistics
    with tab2:
        st.header("Inferential Statistics")
        
        test_type = st.selectbox("Select Test Type:", 
                                ["Independent Samples T-Test", 
                                 "Paired Samples T-Test", 
                                 "One-Way ANOVA"])
        
        if test_type == "Independent Samples T-Test":
            st.subheader("Independent Samples T-Test")
            col1, col2 = st.columns(2)
            
            with col1:
                dep_var = st.selectbox("Dependent Variable:", 
                                      st.session_state.data.select_dtypes(include=[np.number]).columns,
                                      key='ind_dep')
            
            with col2:
                group_var = st.selectbox("Grouping Variable:", 
                                        st.session_state.data.columns,
                                        key='ind_group')
            
            if st.button("Run Independent T-Test"):
                st.session_state.analyzer.clear_output()
                st.session_state.analyzer.independent_t_test(dep_var, group_var)
                output = st.session_state.analyzer.get_output()
                st.code(output, language='text')
        
        elif test_type == "Paired Samples T-Test":
            st.subheader("Paired Samples T-Test")
            col1, col2 = st.columns(2)
            
            numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
            
            with col1:
                var1 = st.selectbox("Variable 1:", numeric_cols, key='pair_var1')
            
            with col2:
                var2 = st.selectbox("Variable 2:", numeric_cols, key='pair_var2')
            
            if st.button("Run Paired T-Test"):
                st.session_state.analyzer.clear_output()
                st.session_state.analyzer.paired_t_test(var1, var2)
                output = st.session_state.analyzer.get_output()
                st.code(output, language='text')
        
        elif test_type == "One-Way ANOVA":
            st.subheader("One-Way ANOVA")
            col1, col2 = st.columns(2)
            
            with col1:
                dep_var_anova = st.selectbox("Dependent Variable:", 
                                            st.session_state.data.select_dtypes(include=[np.number]).columns,
                                            key='anova_dep')
            
            with col2:
                factor_var = st.selectbox("Factor Variable:", 
                                         st.session_state.data.columns,
                                         key='anova_factor')
            
            if st.button("Run One-Way ANOVA"):
                st.session_state.analyzer.clear_output()
                st.session_state.analyzer.one_way_anova(dep_var_anova, factor_var)
                output = st.session_state.analyzer.get_output()
                st.code(output, language='text')
    
    # Tab 3: Correlation
    with tab3:
        st.header("Correlation Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        numeric_cols = st.session_state.data.select_dtypes(include=[np.number]).columns
        
        with col1:
            corr_var1 = st.selectbox("Variable 1:", numeric_cols, key='corr_var1')
        
        with col2:
            corr_var2 = st.selectbox("Variable 2:", numeric_cols, key='corr_var2')
        
        with col3:
            corr_method = st.selectbox("Method:", ["pearson", "spearman"], key='corr_method')
        
        if st.button("Run Correlation"):
            st.session_state.analyzer.clear_output()
            st.session_state.analyzer.correlation(corr_var1, corr_var2, method=corr_method)
            output = st.session_state.analyzer.get_output()
            st.code(output, language='text')
    
    # Tab 4: Advanced
    with tab4:
        st.header("Data Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Types")
            st.dataframe(pd.DataFrame({
                'Variable': st.session_state.data.columns,
                'Type': st.session_state.data.dtypes.astype(str),
                'Non-Null Count': st.session_state.data.count(),
                'Null Count': st.session_state.data.isna().sum()
            }), use_container_width=True)
        
        with col2:
            st.subheader("Quick Statistics")
            st.dataframe(st.session_state.data.describe(), use_container_width=True)
        
        st.subheader("Download Data")
        csv = st.session_state.data.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name="analysis_data.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
