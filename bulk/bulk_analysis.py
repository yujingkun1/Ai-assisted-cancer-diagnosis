import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Store data globally for simplicity (in production, use a database or session)
global_df_target = None
global_df = None
global_result = None

@app.route('/Uploads', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file:
        filename = secure_filename(str(uuid.uuid4()) + '_' + file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        global global_df_target
        if filename.endswith('.xlsx'):
            global_df_target = pd.read_excel(file_path)
        else:
            global_df_target = pd.read_csv(file_path)
        
        global_df_target.columns = ['gene_id'] + list(global_df_target.columns[1:])
        global_df_target.set_index('gene_id', inplace=True)
        
        return jsonify({'message': 'File uploaded successfully'}), 200

@app.route('/visualize', methods=['POST'])
def visualize():
    global global_df_target, global_df, global_result
    try:
        # Load COAD_expression_tpm.csv (assuming it's available in the working directory)
        global_df = pd.read_csv('COAD_expression_tpm.csv')
        global_df.columns = ['gene_id'] + list(global_df.columns[1:])
        global_df.set_index('gene_id', inplace=True)
        
        # Process data as per original code
        df = global_df.copy()
        df = df[df.median(axis=1) > 1]
        global_df_target = global_df_target.loc[df.index.intersection(global_df_target.index)]
        df = df.loc[df.index.intersection(global_df_target.index)]
        df = df.loc[~df.index.duplicated(keep='first')]
        df = df.loc[:, df.columns.str.contains('-11A-')]
        global_df_target = global_df_target.loc[~global_df_target.index.duplicated(keep='first')]
        
        common_genes = df.index.intersection(global_df_target.index)
        df = df.loc[common_genes]
        global_df_target = global_df_target.loc[common_genes]
        
        gene_mean = df.mean(axis=1)
        gene_std = df.std(axis=1)
        
        # Bubble chart data
        samples = global_df_target.columns[0]
        values = global_df_target[samples].sort_values(ascending=False)[:100]
        value = np.log10(values + 1)
        gene_ids = value.index
        
        # Histogram data
        global_df_target.index = global_df_target.index.str[:15]
        df_target_nonzero = global_df_target[global_df_target > 0][samples].dropna()
        
        # Boxplot data
        normal_ranges = pd.DataFrame({
            'Lower': gene_mean - 3 * gene_std,
            'Upper': gene_mean + 3 * gene_std
        })
        target_values = global_df_target.loc[common_genes]
        merged = pd.concat([normal_ranges, target_values], axis=1)
        merged.columns = ['Lower', 'Upper', 'Target Value']
        merged['Abnormality'] = 'Normal'
        merged.loc[merged['Target Value'] < merged['Lower'], 'Abnormality'] = 'Significantly Less'
        merged.loc[merged['Target Value'] > merged['Upper'], 'Abnormality'] = 'Significantly More'
        abnormal_genes = merged[merged['Abnormality'] != 'Normal']
        global_result = abnormal_genes.reset_index()
        global_result.columns = ['Gene', 'Lower', 'Upper', 'Target Value', 'Abnormality']
        global_result['Normal Range'] = list(zip(global_result['Lower'], global_result['Upper']))
        global_result = global_result[['Gene', 'Normal Range', 'Target Value', 'Abnormality']]
        global_result = global_result.head(50)
        
        # Generate static plots (for compatibility with existing functionality)
        plt.figure(figsize=(6, 20))
        colors = plt.cm.viridis_r(np.linspace(0, 1, len(value)))
        plt.scatter(value, gene_ids, s=value * 10, alpha=0.8, c=colors)
        plt.colorbar(label='log10(TPM)')
        plt.gca().invert_yaxis()
        plt.title('Top 100 genes in sample')
        plt.xlabel('log10(TPM)')
        plt.ylabel('Gene ID')
        plt.savefig(os.path.join(OUTPUT_FOLDER, 'top_100_genes.png'), dpi=300)
        plt.close()
        
        plt.figure(figsize=(8, 6))
        plt.hist(df_target_nonzero, bins=100, log=True)
        plt.title('Histogram of gene expression values')
        plt.xlabel('Expression value')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(OUTPUT_FOLDER, 'gene_expression_histogram.png'), dpi=300)
        plt.close()
        
        fig, ax = plt.subplots(figsize=(20, 12))
        genes = global_result['Gene'].tolist()
        normal_ranges = global_result['Normal Range'].tolist()
        target_values = global_result['Target Value'].tolist()
        for i, gene in enumerate(genes):
            ax.broken_barh([(normal_ranges[i][0], normal_ranges[i][1] - normal_ranges[i][0])], (i - 0.4, 0.8), facecolors='lightblue')
            ax.plot(target_values[i], i, 'ro', markersize=8, label='Target Value' if i == 0 else "")
        ax.set_yticks(range(len(genes)))
        ax.set_yticklabels(genes)
        ax.set_xlabel('Value')
        ax.set_title('Box Plot with Normal Range and Target Value')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, 'abnormal_genes_boxplot.png'), dpi=300)
        plt.close()
        
        return jsonify({'message': 'Visualizations generated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_visualization_data', methods=['GET'])
def get_visualization_data():
    global global_df_target, global_df, global_result
    try:
        if global_df_target is None or global_df is None:
            return jsonify({'error': 'No data available. Please upload a file first.'}), 400
        
        # Bubble chart data
        samples = global_df_target.columns[0]
        values = global_df_target[samples].sort_values(ascending=False)[:100]
        value = np.log10(values + 1)
        bubble_data = {
            'x': value.tolist(),
            'y': value.index.tolist(),
            'size': (value * 10).tolist(),
            'color': value.tolist()
        }
        
        # Histogram data
        df_target_nonzero = global_df_target[global_df_target > 0][samples].dropna()
        hist_data = df_target_nonzero.tolist()
        
        # Boxplot data
        boxplot_data = {
            'genes': global_result['Gene'].tolist(),
            'normal_ranges': global_result['Normal Range'].tolist(),
            'target_values': global_result['Target Value'].tolist()
        }
        
        return jsonify({
            'bubble': bubble_data,
            'histogram': hist_data,
            'boxplot': boxplot_data
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)