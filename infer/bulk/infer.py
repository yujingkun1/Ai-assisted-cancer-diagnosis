import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def analyze_expression(input_path, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Read data
    logger.debug("Reading input file: %s", input_path)
    df_target = pd.read_excel(input_path)
    df_target.columns = ['gene_id'] + list(df_target.columns[1:])
    df_target.set_index('gene_id', inplace=True)
    logger.debug("Input data shape: %s", df_target.shape)
    
    # Bubble chart (Top 50 genes)
    sample = df_target.columns[0]
    values = df_target[sample].sort_values(ascending=False)[:50]
    log_values = np.log10(values + 1)
    gene_ids = values.index
    logger.debug("Top 50 genes selected, length: %d", len(gene_ids))
    
    plt.figure(figsize=(6, 20))
    colors = plt.cm.viridis_r(np.linspace(0, 1, len(log_values)))
    plt.scatter(log_values, values.index, s=log_values*5, c=colors, alpha=0.7)
    plt.title('Top 50 Expressed Genes')
    plt.xlabel('log10(TPM+1)')
    plt.ylabel('Gene ID')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'top_50_genes.png'), dpi=300)
    plt.close()
    
    # Histogram
    df_target.index = df_target.index.str[:15]
    non_zero = df_target[df_target > 0][sample].dropna().values.flatten()
    logger.debug("Non-zero values for histogram: %d", len(non_zero))
    
    plt.figure(figsize=(10, 6))
    plt.hist(non_zero, bins=50, log=True, color='skyblue', edgecolor='black')
    plt.title('Gene Expression Distribution')
    plt.xlabel('TPM')
    plt.ylabel('Frequency (log scale)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'gene_histogram.png'), dpi=300)
    plt.close()
    
    # Load reference data
    logger.debug("Loading reference data: bulk/COAD_expression_tpm.csv")
    df = pd.read_csv(r'bulk/COAD_expression_tpm.csv')
    df.columns = ['gene_id'] + list(df.columns[1:])
    df.set_index('gene_id', inplace=True)
    df = df[df.median(axis=1) > 1]
    df_target = df_target.loc[df.index.intersection(df_target.index)]
    df = df.loc[df.index.intersection(df_target.index)]
    df = df.loc[~df.index.duplicated(keep='first')]
    df = df.loc[:, df.columns.str.contains('-11A-')]
    df_target = df_target.loc[~df_target.index.duplicated(keep='first')]
    
    # Data processing for boxplot
    common_genes = df.index.intersection(df_target.index)
    logger.debug("Common genes count: %d", len(common_genes))
    if len(common_genes) == 0:
        logger.warning("No common genes found between input and reference data")
        return {
            'bubble': {
                'x': log_values.tolist(),
                'y': gene_ids.tolist(),
                'size': (log_values * 5).tolist(),
                'color': log_values.tolist()
            },
            'histogram': non_zero.tolist(),
            'boxplot': {
                'genes': [],
                'normal_ranges': [],
                'target_values': []
            }
        }
    
    df = df.loc[common_genes]
    df_target = df_target.loc[common_genes]
    
    gene_mean = df.mean(axis=1)
    gene_std = df.std(axis=1)
    
    normal_ranges = pd.DataFrame({
        'Lower': gene_mean - 3 * gene_std,
        'Upper': gene_mean + 3 * gene_std
    })
    
    target_values = df_target.loc[common_genes]
    
    merged = pd.concat([normal_ranges, target_values], axis=1)
    merged.columns = ['Lower', 'Upper', 'Target Value']
    
    merged['Abnormality'] = 'Normal'
    merged.loc[merged['Target Value'] < merged['Lower'], 'Abnormality'] = 'Significantly Less'
    merged.loc[merged['Target Value'] > merged['Upper'], 'Abnormality'] = 'Significantly More'
    
    abnormal_genes = merged[merged['Abnormality'] != 'Normal']
    logger.debug("Abnormal genes count: %d", len(abnormal_genes))
    
    result = abnormal_genes.reset_index()
    result.columns = ['Gene', 'Lower', 'Upper', 'Target Value', 'Abnormality']
    result['Normal Range'] = list(zip(result['Lower'], result['Upper']))
    result = result[['Gene', 'Normal Range', 'Target Value', 'Abnormality']]
    
    # Save CSV
    result.to_csv(os.path.join(output_folder, 'abnormal_genes.csv'), index=False)
    
    # Boxplot (limit to top 50 genes)
    result = result.head(50)
    genes = result['Gene'].tolist()
    normal_ranges_list = result['Normal Range'].tolist()
    target_values_list = result['Target Value'].tolist()
    logger.debug("Boxplot genes count: %d", len(genes))
    
    # Clean NaN values
    clean_normal_ranges = []
    clean_target_values = []
    clean_genes = []
    for i, (gene, rng, val) in enumerate(zip(genes, normal_ranges_list, target_values_list)):
        if (not np.isnan(rng[0]) and not np.isnan(rng[1]) and 
            not np.isnan(val) and val is not None):
            clean_normal_ranges.append(rng)
            clean_target_values.append(val)
            clean_genes.append(gene)
        else:
            logger.warning("Skipping gene %s due to NaN or invalid values: range=%s, target=%s", 
                         gene, rng, val)
    
    logger.debug("Cleaned boxplot genes count: %d", len(clean_genes))
    logger.debug("Sample normal_ranges: %s", clean_normal_ranges[:5])
    logger.debug("Sample target_values: %s", clean_target_values[:5])
    
    # Generate static boxplot
    if clean_genes:
        fig, ax = plt.subplots(figsize=(20, 12))
        for i, gene in enumerate(clean_genes):
            ax.broken_barh([(clean_normal_ranges[i][0], 
                           clean_normal_ranges[i][1] - clean_normal_ranges[i][0])], 
                          (i - 0.4, 0.8), facecolors='lightblue')
            ax.plot(clean_target_values[i], i, 'ro', markersize=8, 
                   label='Target Value' if i == 0 else "")
        
        ax.set_yticks(range(len(clean_genes)))
        ax.set_yticklabels(clean_genes)
        ax.set_xlabel('Value')
        ax.set_title('Box Plot with Normal Range and Target Value')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'abnormal_genes_boxplot.png'), dpi=300)
        plt.close()
    
    # Prepare data for Plotly
    plotly_data = {
        'bubble': {
            'x': log_values.tolist(),
            'y': gene_ids.tolist(),
            'size': (log_values * 5).tolist(),
            'color': log_values.tolist()
        },
        'histogram': non_zero.tolist(),
        'boxplot': {
            'genes': clean_genes,
            'normal_ranges': clean_normal_ranges,
            'target_values': clean_target_values
        }
    }
    
    return plotly_data