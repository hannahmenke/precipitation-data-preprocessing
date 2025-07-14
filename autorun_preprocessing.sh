#!/bin/bash

# Precipitation Data Preprocessing Pipeline
# This script automates the complete workflow from BMP files to filtered TIFF files
# 
# Steps:
# 1. Activate conda environment
# 2. Convert BMP files to TIFF format
# 3. Apply non-local means filtering with Avizo-matched settings
# 4. Generate summary report

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONDA_ENV_NAME="precipitation_data"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="preprocessing_$(date +%Y%m%d_%H%M%S).log"

# Function to print colored messages
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

# Function to check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed or not in PATH"
        print_error "Please install Anaconda/Miniconda and try again"
        exit 1
    fi
    print_success "Conda found: $(conda --version)"
}

# Function to check if environment exists
check_environment() {
    if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
        print_warning "Environment '${CONDA_ENV_NAME}' not found"
        print_status "Creating conda environment from environment.yml..."
        
        if [ -f "environment.yml" ]; then
            conda env create -f environment.yml
            print_success "Environment created successfully"
        else
            print_error "environment.yml not found. Creating environment manually..."
            conda create -n "${CONDA_ENV_NAME}" python=3.11 numpy=1.26 pillow matplotlib opencv -y
            print_success "Environment created manually"
        fi
    else
        print_success "Environment '${CONDA_ENV_NAME}' found"
    fi
}

# Function to activate conda environment
activate_environment() {
    print_status "Activating conda environment: ${CONDA_ENV_NAME}"
    
    # Initialize conda for bash
    eval "$(conda shell.bash hook)"
    
    # Activate the environment
    conda activate "${CONDA_ENV_NAME}"
    
    # Verify activation
    if [[ "$CONDA_DEFAULT_ENV" == "${CONDA_ENV_NAME}" ]]; then
        print_success "Environment activated successfully"
        print_status "Python version: $(python --version)"
        print_status "Working directory: $(pwd)"
    else
        print_error "Failed to activate environment"
        exit 1
    fi
}

# Function to run BMP to TIFF conversion
run_bmp_conversion() {
    print_status "=== STEP 1: Converting BMP files to TIFF format ==="
    
    if [ ! -f "bmp_to_tiff_converter.py" ]; then
        print_error "bmp_to_tiff_converter.py not found!"
        exit 1
    fi
    
    # Count BMP files first
    bmp_count=$(find . -name "*.bmp" -type f | wc -l)
    print_status "Found ${bmp_count} BMP file(s) to process"
    
    if [ "$bmp_count" -eq 0 ]; then
        print_warning "No BMP files found. Skipping conversion step."
        return 0
    fi
    
    # Build command with options
    bmp_cmd="python bmp_to_tiff_converter.py"
    if [ "$FORCE_FLAG" = "--force" ]; then
        bmp_cmd="$bmp_cmd --force"
    fi
    if [ -n "$GRAYSCALE_FLAG" ]; then
        bmp_cmd="$bmp_cmd $GRAYSCALE_FLAG"
        print_status "Converting to grayscale"
    fi
    if [ -n "$CHANNEL_FLAG" ]; then
        bmp_cmd="$bmp_cmd $CHANNEL_FLAG"
        print_status "Extracting specific channel"
    fi
    
    # Run the conversion
    print_status "Running BMP to TIFF converter..."
    if eval "$bmp_cmd"; then
        print_success "BMP to TIFF conversion completed successfully"
    else
        print_error "BMP to TIFF conversion failed"
        exit 1
    fi
}

# Function to run non-local means filtering
run_nlm_filtering() {
    print_status "=== STEP 2: Applying Non-Local Means Filtering ==="
    
    if [ ! -f "nonlocal_means_filter.py" ]; then
        print_error "nonlocal_means_filter.py not found!"
        exit 1
    fi
    
    # Count TIFF files first
    tiff_count=$(find . -name "*.tiff" -not -name "*_nlm_filtered.tiff" -type f | wc -l)
    print_status "Found ${tiff_count} TIFF file(s) to filter"
    
    if [ "$tiff_count" -eq 0 ]; then
        print_warning "No TIFF files found. Skipping filtering step."
        return 0
    fi
    
    # Run the filtering with Avizo-matched settings
    print_status "Running Non-Local Means filter with Avizo settings:"
    print_status "  - Search window: 10 px"
    print_status "  - Local neighborhood: 3 pixels"
    print_status "  - Similarity value: 0.8 (h=6.0)"
    
    if python nonlocal_means_filter.py; then
        print_success "Non-local means filtering completed successfully"
    else
        print_error "Non-local means filtering failed"
        exit 1
    fi
}

# Function to generate summary report
generate_summary() {
    print_status "=== PROCESSING SUMMARY ==="
    
    # Count files
    bmp_files=$(find . -name "*.bmp" -type f | wc -l)
    tiff_files=$(find . -name "*.tiff" -not -name "*_nlm_filtered.tiff" -type f | wc -l)
    filtered_files=$(find . -name "*_nlm_filtered.tiff" -type f | wc -l)
    
    print_status "File inventory:"
    print_status "  - BMP files: ${bmp_files}"
    print_status "  - Raw TIFF files: ${tiff_files}"
    print_status "  - Filtered TIFF files: ${filtered_files}"
    
    # Calculate total file sizes
    if command -v du &> /dev/null; then
        total_size=$(du -sh . 2>/dev/null | cut -f1 || echo "Unknown")
        print_status "  - Total directory size: ${total_size}"
    fi
    
    # Processing time
    if [ -n "$START_TIME" ]; then
        end_time=$(date +%s)
        duration=$((end_time - START_TIME))
        hours=$((duration / 3600))
        minutes=$(((duration % 3600) / 60))
        seconds=$((duration % 60))
        
        if [ $hours -gt 0 ]; then
            time_str="${hours}h ${minutes}m ${seconds}s"
        elif [ $minutes -gt 0 ]; then
            time_str="${minutes}m ${seconds}s"
        else
            time_str="${seconds}s"
        fi
        
        print_status "  - Total processing time: ${time_str}"
    fi
    
    print_status "Log file saved as: ${LOG_FILE}"
}

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --skip-bmp         Skip BMP to TIFF conversion"
    echo "  --skip-filter      Skip non-local means filtering"
    echo "  --force            Force reprocessing of existing files"
    echo "  --pattern PATTERN  Process only files matching pattern"
    echo "  --grayscale        Convert BMP files to grayscale TIFF"
    echo "  --channel N        Extract specific channel (0=R, 1=G, 2=B, 3=A, 4=A or G fallback)"
    echo "  --in-memory        Use in-memory workflow (BMP -> filtered TIFF, no intermediate files)"
    echo "  --help, -h         Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                          # Run complete pipeline"
    echo "  $0 --skip-bmp               # Only run filtering"
    echo "  $0 --pattern '6mM'          # Process only 6mM files"
    echo "  $0 --force                  # Reprocess all files"
    echo "  $0 --grayscale              # Convert BMP to grayscale TIFF"
    echo "  $0 --channel 4              # Extract channel 4 from BMP files"
    echo "  $0 --in-memory              # Use memory-efficient workflow (no intermediate TIFFs)"
    echo "  $0 --grayscale --pattern '3mM'  # Grayscale conversion for 3mM files only"
}

# Parse command line arguments
SKIP_BMP=false
SKIP_FILTER=false
FORCE_FLAG=""
PATTERN_FLAG=""
GRAYSCALE_FLAG=""
CHANNEL_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-bmp)
            SKIP_BMP=true
            shift
            ;;
        --skip-filter)
            SKIP_FILTER=true
            shift
            ;;
        --force)
            FORCE_FLAG="--force"
            shift
            ;;
        --pattern)
            PATTERN_FLAG="--pattern $2"
            shift 2
            ;;
        --grayscale)
            GRAYSCALE_FLAG="--grayscale"
            shift
            ;;
        --channel)
            CHANNEL_FLAG="--channel $2"
            shift 2
            ;;
        --in-memory)
            IN_MEMORY=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    START_TIME=$(date +%s)
    
    print_status "Starting Precipitation Data Preprocessing Pipeline"
    print_status "Timestamp: $(date)"
    print_status "Working directory: $(pwd)"
    print_status "Log file: ${LOG_FILE}"
    echo ""
    
    # Check prerequisites
    check_conda
    check_environment
    activate_environment
    
    echo ""
    
    # Run processing steps
    if [ "$IN_MEMORY" = true ]; then
        print_status "Using in-memory workflow (BMP -> filtered TIFF, no intermediate files)"
        
        # Build command for in-memory workflow
        memory_cmd="python bmp_to_filtered_workflow.py 3mM 6mM"
        
        if [ -n "$FORCE_FLAG" ]; then
            memory_cmd="$memory_cmd --force"
        fi
        
        if [ -n "$PATTERN_FLAG" ]; then
            memory_cmd="$memory_cmd $PATTERN_FLAG"
        fi
        
        if [ -n "$GRAYSCALE_FLAG" ]; then
            memory_cmd="$memory_cmd --grayscale"
        fi
        
        if [ -n "$CHANNEL_FLAG" ]; then
            memory_cmd="$memory_cmd $CHANNEL_FLAG"
        fi
        
        print_status "Running: $memory_cmd"
        if $memory_cmd; then
            print_success "In-memory workflow completed successfully"
        else
            print_error "In-memory workflow failed"
            exit 1
        fi
        echo ""
    else
        # Traditional two-step workflow
        if [ "$SKIP_BMP" = false ]; then
            run_bmp_conversion
            echo ""
        else
            print_warning "Skipping BMP to TIFF conversion (--skip-bmp flag used)"
            echo ""
        fi
        
        if [ "$SKIP_FILTER" = false ]; then
            # Add force and pattern flags if specified
            if [ -n "$FORCE_FLAG" ] || [ -n "$PATTERN_FLAG" ]; then
                print_status "Additional flags: ${FORCE_FLAG} ${PATTERN_FLAG}"
                if python nonlocal_means_filter.py $FORCE_FLAG $PATTERN_FLAG; then
                    print_success "Non-local means filtering completed successfully"
                else
                    print_error "Non-local means filtering failed"
                    exit 1
                fi
            else
                run_nlm_filtering
            fi
            echo ""
        else
            print_warning "Skipping non-local means filtering (--skip-filter flag used)"
            echo ""
        fi
    fi
    
    # Generate summary
    generate_summary
    
    print_success "=== PIPELINE COMPLETED SUCCESSFULLY ==="
    print_status "Next steps:"
    print_status "  - Use raw_vs_filtered_inspector.py to compare results"
    print_status "  - Use image_quality_inspector.py to compare BMP vs TIFF quality"
}

# Trap to ensure we always generate a summary, even on failure
trap 'generate_summary; exit 1' ERR

# Run the main function
main "$@" 