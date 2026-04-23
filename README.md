# TrackMyPDB - Streamlit Application

**A comprehensive bioinformatics pipeline for extracting heteroatoms from protein structures and finding molecularly similar compounds using fingerprint-based similarity analysis.**

© 2025 [Standard Seed Corporation](https://www.linkedin.com/company/standard-seed-corporation/?viewAsMember=true). This is an open-source project developed and released by Standard Seed Corporation under the MIT License. All rights reserved.


## 🎯 Overview

TrackMyPDB is a user-friendly Streamlit web application that combines two powerful components:

1. **Heteroatom Extraction Tool**: Systematically extracts all heteroatoms from PDB structures associated with UniProt proteins
2. **Molecular Similarity Analyzer**: Finds ligands most similar to a target molecule using Morgan fingerprints and Tanimoto similarity

## 🚀 Quick Start

### Web Application
- https://trackmypdbsscai.streamlit.app/


### Launch using local machine
### Prerequisites
- Python 3.7+
- Internet connection for API calls
- Windows OS (optimized for Windows environment)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd TrackMyPDB
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application**:
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser** to `http://localhost:8501`

### Basic Usage

1. **Navigate** to the web interface
2. **Choose analysis type**:
   - 🔍 Heteroatom Extraction
   - 🧪 Similarity Analysis  
   - 📊 Complete Pipeline
3. **Input your data**:
   - UniProt IDs (e.g., Q9UNQ0, P37231, P06276)
   - Target SMILES structure
4. **Run analysis** and download CSV results

## 📋 Application Features

### 🔍 Heteroatom Extraction
- **Input**: UniProt protein identifiers
- **Process**: Fetches PDB structures, extracts heteroatoms, retrieves SMILES
- **Output**: Comprehensive CSV with chemical information
- **APIs**: RCSB PDB, PubChem integration
- **Features**: Progress tracking, error handling, result caching

### 🧪 Molecular Similarity Analysis
- **Input**: Target SMILES structure
- **Process**: Morgan fingerprint computation, Tanimoto similarity calculation
- **Output**: Ranked similarity results with interactive visualizations
- **Features**: Configurable parameters, real-time analysis, comprehensive reports

### 📊 Complete Pipeline
- **Workflow**: End-to-end processing from UniProt IDs to similarity results
- **Integration**: Automatic heteroatom extraction followed by similarity analysis
- **Output**: Both heteroatom database and similarity results

## 🏗️ Project Structure

```
TrackMyPDB/
├── streamlit_app.py              # Main Streamlit application
├── requirements.txt              # Python dependencies
├── backend/
│   ├── __init__.py              # Package initialization
│   ├── heteroatom_extractor.py  # Heteroatom extraction logic
│   └── similarity_analyzer.py   # Similarity analysis logic
└── README.md                    # This file
```

## 🛠️ Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **RDKit**: Cheminformatics and molecular similarity
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Requests**: API communications
- **NumPy**: Numerical computations

### API Integration
- **PDBe REST API**: PDB structure mappings
- **RCSB PDB API**: Chemical component data
- **PubChem API**: Backup molecular data

### Molecular Analysis
- **Morgan Fingerprints**: Circular molecular fingerprints (radius=2, 2048 bits)
- **Tanimoto Similarity**: Industry-standard similarity metric (0-1 scale)
- **Interactive Visualizations**: Distribution plots, similarity rankings, statistical analysis

## 🎮 User Interface

### Apple-Inspired Design
- **Modern UI**: Clean, minimalist design inspired by Apple Design principles
- **Responsive Layout**: Optimized for different screen sizes
- **Interactive Elements**: Smooth animations and hover effects
- **Intuitive Navigation**: Clear section organization and progress indicators

### Key Features
- **Real-time Progress**: Progress bars and status updates
- **Error Handling**: Graceful error messages and troubleshooting
- **Data Export**: CSV download functionality with timestamps
- **Result Caching**: Session state management for efficiency

## 📊 Expected Results

### Typical Output
- **Heteroatoms**: ~1000-5000 heteroatoms per 10 UniProt proteins
- **SMILES Success**: ~60-80% success rate for SMILES retrieval
- **Similar Ligands**: ~50-200 similar compounds per target (similarity > 0.2)
- **Processing Time**: 30-60 minutes for complete pipeline

### File Outputs
- `heteroatom_results_YYYYMMDD_HHMMSS.csv`: Complete heteroatom extraction results
- `similarity_results_YYYYMMDD_HHMMSS.csv`: Molecular similarity analysis results

## 🔧 Configuration Options

### Heteroatom Extraction
- **UniProt IDs**: Multiple input formats (comma-separated, line-separated)
- **Result Caching**: Previous results loading and management
- **API Settings**: Automatic retry logic and rate limiting

### Similarity Analysis
- **Fingerprint Parameters**: 
  - Morgan radius: 1, 2, 3 (default: 2)
  - Fingerprint bits: 1024, 2048, 4096 (default: 2048)
- **Analysis Parameters**:
  - Top N results: 10-100 (default: 50)
  - Minimum similarity: 0.0-1.0 (default: 0.2)

## 🚨 Troubleshooting

### Common Issues

#### "Module not found" errors
```bash
# Install dependencies
pip install -r requirements.txt

# For RDKit installation issues on Windows
conda install -c conda-forge rdkit
```

#### "Invalid SMILES" warnings
- Verify SMILES syntax using online validators
- Check for special characters or formatting issues
- Example valid SMILES: `CCO` (ethanol), `CC(=O)O` (acetic acid)

#### Slow performance
- Reduce number of UniProt IDs for testing
- Use higher minimum similarity threshold
- Check internet connection stability

#### API timeout errors
- Wait a few minutes and retry
- Check if external APIs (RCSB, PubChem) are accessible
- Reduce batch size for large datasets

## 💡 Use Cases

### Drug Discovery
- **Lead Optimization**: Find similar compounds to known drugs
- **Scaffold Hopping**: Identify alternative molecular frameworks
- **Target Analysis**: Understand ligand binding preferences

### Chemical Biology
- **Cofactor Analysis**: Study enzyme cofactor preferences
- **Binding Site Analysis**: Characterize pocket properties
- **Cross-reactivity Prediction**: Assess off-target binding

### Academic Research
- **Structural Biology**: Build custom screening libraries
- **Comparative Analysis**: Study protein-ligand interactions
- **Database Construction**: Create specialized molecular databases

## 🤝 Contributing

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comprehensive error handling
- Include progress indicators for long operations
- Document all functions and classes
- Test with various input formats

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Open Source Project** - Free to use, modify, and distribute under the MIT License terms.

Please respect API terms of service and rate limits when using this application.

## 🙏 Acknowledgments

- **RCSB PDB**: Protein structure data
- **PDBe**: Structure mapping services  
- **PubChem**: Chemical information database
- **RDKit**: Cheminformatics toolkit
- **Streamlit**: Web application framework

---

## 👨‍💻 Developers

- Prject Lead/Senior Engineer: [Sul sharif](https://www.linkedin.com/in/sulimansharif/)
- Lead Engineer: [Anu Gamage](https://www.linkedin.com/in/anu-gamage-62192b201/?originalSubdomain=lk)
- Associate Engineers: [Damilola Bodun](https://www.linkedin.com/in/damilola-bodun-123987208/), [Kalana Kotawalagedara](https://www.linkedin.com/in/kalana-kotawalagedara-962939225/), [Sakeer Sha](https://www.linkedin.com/in/mohammadhu-rifaydheen-sakeer/), [Pamalka Hansani](https://www.linkedin.com/in/phkaldera/) & [Logan Geffen](https://www.linkedin.com/in/logan-geffen/)

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Verify input data format
3. Test with provided examples
4. Review browser console for errors
5. Contact the developers through LinkedIn

**Happy molecular hunting!** 🧬🔍
